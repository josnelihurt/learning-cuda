package websocket

import (
	"context"
	"errors"
	"net/http"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"github.com/jrb/cuda-learning/webserver/pkg/config"
	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	imageinfra "github.com/jrb/cuda-learning/webserver/pkg/infrastructure/image"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/logger"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/video"
	"github.com/jrb/cuda-learning/webserver/pkg/interfaces/adapters"
	"github.com/jrb/cuda-learning/webserver/pkg/telemetry"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

const (
	transportFormatBinary = "binary"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

// connID is a unique identifier for WebSocket connections
// Using a string ID instead of pointer as map key prevents runtime panics
type connID string

type Handler struct {
	useCase             *application.ProcessImageUseCase
	evaluateFFUse       *application.EvaluateFeatureFlagUseCase
	grpcProcessor       domain.ImageProcessor
	imageCodec          *imageinfra.Codec
	adapter             *adapters.ProtobufAdapter
	streamConfig        config.StreamConfig
	frameCounter        int
	videoSessionManager *VideoSessionManager
	videoRepository     domain.VideoRepository
	// Fixed: Use connID (string) as key instead of pointer to prevent runtime panics
	connMutexes     map[connID]*sync.Mutex
	connMutexesLock sync.Mutex
	// Track connections by ID for cleanup
	connections   map[connID]*websocket.Conn
	connectionsMu sync.RWMutex
	// Track active goroutines for graceful shutdown
	activeGoroutines sync.WaitGroup
	// Track goroutine count for monitoring
	activeGoroutineCount int64
}

func NewHandler(
	useCase *application.ProcessImageUseCase,
	streamCfg config.StreamConfig,
	videoRepo domain.VideoRepository,
	evaluateFFUse *application.EvaluateFeatureFlagUseCase,
	grpcProcessor domain.ImageProcessor,
) *Handler {
	return &Handler{
		useCase:             useCase,
		evaluateFFUse:       evaluateFFUse,
		grpcProcessor:       grpcProcessor,
		imageCodec:          imageinfra.NewImageCodec(),
		adapter:             adapters.NewProtobufAdapter(),
		streamConfig:        streamCfg,
		frameCounter:        0,
		videoSessionManager: NewVideoSessionManager(),
		videoRepository:     videoRepo,
		connMutexes:         make(map[connID]*sync.Mutex),
		connections:         make(map[connID]*websocket.Conn),
	}
}

// safeWriteMessage writes to WebSocket with mutex protection to prevent concurrent write panics
// Fixed: Now uses connID instead of pointer as map key to prevent runtime panics
func (h *Handler) safeWriteMessage(id connID, messageType int, data []byte) error {
	h.connMutexesLock.Lock()
	mu, exists := h.connMutexes[id]
	if !exists {
		mu = &sync.Mutex{}
		h.connMutexes[id] = mu
	}
	h.connMutexesLock.Unlock()

	mu.Lock()
	defer mu.Unlock()

	h.connectionsMu.RLock()
	conn, ok := h.connections[id]
	h.connectionsMu.RUnlock()

	if !ok || conn == nil {
		return errors.New("connection not found")
	}

	return conn.WriteMessage(messageType, data)
}

// registerConnection registers a new connection and returns its unique ID
func (h *Handler) registerConnection(conn *websocket.Conn) connID {
	id := connID(time.Now().Format("20060102-150405.000000") + "-" + randomString(8))
	h.connectionsMu.Lock()
	h.connections[id] = conn
	h.connectionsMu.Unlock()

	// Initialize mutex for this connection
	h.connMutexesLock.Lock()
	if _, exists := h.connMutexes[id]; !exists {
		h.connMutexes[id] = &sync.Mutex{}
	}
	h.connMutexesLock.Unlock()

	return id
}

// cleanupConnection removes a connection and its mutex
func (h *Handler) cleanupConnection(id connID) {
	h.connMutexesLock.Lock()
	delete(h.connMutexes, id)
	h.connMutexesLock.Unlock()

	h.connectionsMu.Lock()
	delete(h.connections, id)
	h.connectionsMu.Unlock()
}

// randomString generates a random string for connection IDs
func randomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, length)
	for i := range b {
		b[i] = charset[time.Now().UnixNano()%int64(len(charset))]
	}
	return string(b)
}

// TODO: To be replaced by gRPC bidirectional streaming
// Target replacement: webserver/pkg/interfaces/connectrpc/handler.go StreamProcessVideo method
// Migration path: Implement StreamProcessVideo RPC, update frontend to use Connect-Web streaming
// Benefits: Type-safe protocol, unified API surface, better error handling, no custom WebSocket logic
// Keep this during migration for backward compatibility with existing clients
func (h *Handler) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
	ctx := telemetry.ExtractFromHTTPHeaders(r.Context(), r.Header)
	tracer := otel.Tracer("websocket-handler")
	log := logger.FromContext(ctx)

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Error().Err(err).Msg("WebSocket upgrade failed")
		return
	}

	// Fixed: Register connection and get unique ID instead of using pointer as key
	registeredConnID := h.registerConnection(conn)
	defer func() {
		h.cleanupConnection(registeredConnID)
		conn.Close()
	}()

	transportFormat := h.streamConfig.TransportFormat
	log.Info().
		Str("transport_format", transportFormat).
		Str("conn_id", string(registeredConnID)).
		Msg("WebSocket connected")

	for {
		messageType, message, err := conn.ReadMessage()
		if err != nil {
			log.Info().Err(err).Msg("ReadMessage error in handler loop")
			break
		}

		log.Info().Int("message_size", len(message)).Int("message_type", messageType).Msg("Received WebSocket message")

		// Detect transport format from message type: BinaryMessage = binary, TextMessage = json
		// This allows clients to specify format dynamically even if server config differs
		detectedFormat := transportFormat
		if messageType == websocket.BinaryMessage {
			detectedFormat = transportFormatBinary
		} else if messageType == websocket.TextMessage {
			detectedFormat = "json"
		}

		var frameMsg pb.WebSocketFrameRequest
		if detectedFormat == transportFormatBinary {
			err = proto.Unmarshal(message, &frameMsg)
		} else {
			unmarshalOptions := protojson.UnmarshalOptions{
				DiscardUnknown: false,
			}
			err = unmarshalOptions.Unmarshal(message, &frameMsg)
		}

		if err != nil {
			log.Error().Err(err).Str("message", string(message)).Msg("Failed to unmarshal message")
			continue
		}

		log.Info().Str("msg_type", frameMsg.Type).Bool("has_start_video_req", frameMsg.StartVideoRequest != nil).Msg("Processing WebSocket message")

		if frameMsg.Type == "start_video" {
			log.Info().Str("video_id", frameMsg.StartVideoRequest.GetVideoId()).Msg("Received start_video message")
			h.handleStartVideo(ctx, registeredConnID, conn, &frameMsg, messageType)
			continue
		}

		if frameMsg.Type == "stop_video" {
			log.Info().Str("session_id", frameMsg.StopVideoRequest.GetSessionId()).Msg("Received stop_video message")
			h.handleStopVideo(ctx, &frameMsg)
			continue
		}

		result := h.processFrame(ctx, tracer, &frameMsg)

		var responseBytes []byte
		// Use detected format for response to match request format
		if detectedFormat == transportFormatBinary {
			responseBytes, err = proto.Marshal(result)
		} else {
			responseBytes, err = protojson.Marshal(result)
		}

		if err != nil {
			log.Error().Err(err).Msg("Failed to marshal response")
			continue
		}

		// Fixed: Use registeredConnID instead of conn pointer
		if err := h.safeWriteMessage(registeredConnID, messageType, responseBytes); err != nil {
			break
		}
	}
}

func (h *Handler) handleStartVideo(ctx context.Context, connID connID, conn *websocket.Conn, frameMsg *pb.WebSocketFrameRequest, _ int) {
	log := logger.FromContext(ctx)

	if frameMsg.StartVideoRequest == nil {
		log.Error().Msg("Missing start_video_request in message")
		return
	}

	req := frameMsg.StartVideoRequest
	videoID := req.VideoId

	video, err := h.videoRepository.GetByID(ctx, videoID)
	if err != nil {
		log.Error().Err(err).Str("video_id", videoID).Msg("Failed to get video")
		return
	}

	sessionID := videoID + "-" + time.Now().Format("20060102-150405")
	// Fixed: Pass parent context instead of discarding it
	session := h.videoSessionManager.CreateSession(ctx, sessionID, videoID, connID, conn, "json")

	// Convert web path "/data/videos/filename.mp4" to filesystem path "data/videos/filename.mp4"
	realPath := filepath.Join("data", "videos", filepath.Base(video.Path))

	log.Info().Str("session_id", sessionID).Str("video_path", realPath).Str("video_id", videoID).Msg("Starting video playback")

	// Fixed: Track goroutine with WaitGroup to prevent leaks
	h.activeGoroutines.Add(1)
	atomic.AddInt64(&h.activeGoroutineCount, 1)
	go func() {
		defer h.activeGoroutines.Done()
		defer atomic.AddInt64(&h.activeGoroutineCount, -1)
		h.streamVideoFrames(ctx, session, realPath, req)
	}()
}

func (h *Handler) handleStopVideo(ctx context.Context, frameMsg *pb.WebSocketFrameRequest) {
	log := logger.FromContext(ctx)

	if frameMsg.StopVideoRequest == nil {
		log.Error().Msg("Missing stop_video_request in message")
		return
	}

	sessionID := frameMsg.StopVideoRequest.SessionId
	session := h.videoSessionManager.GetSession(sessionID)
	if session != nil {
		session.CancelFunc()
		h.videoSessionManager.RemoveSession(sessionID)
		log.Info().Str("session_id", sessionID).Msg("Video playback stopped")
	}
}

func (h *Handler) streamVideoFrames(parentCtx context.Context, session *VideoSession, videoPath string, req *pb.StartVideoPlaybackRequest) {
	ctx, cancel := context.WithCancel(parentCtx)
	session.CancelFunc = cancel
	defer cancel()

	log := logger.FromContext(ctx)
	tracer := otel.Tracer("websocket-video-streamer")
	_, span := tracer.Start(ctx, "StreamVideoFrames")
	defer span.End()

	span.SetAttributes(
		attribute.String("video.path", videoPath),
		attribute.String("session.id", session.ID),
	)

	previewPath := filepath.Join(filepath.Dir(videoPath), "..", "video_previews", filepath.Base(videoPath)+".png")

	log.Info().Str("video_path", videoPath).Str("preview", previewPath).Msg("Starting real video streaming with FFmpeg")

	h.streamRealVideo(ctx, session, videoPath, req)
}

func (h *Handler) processFrame(ctx context.Context, tracer trace.Tracer, frameMsg *pb.WebSocketFrameRequest) *pb.WebSocketFrameResponse {
	ctx = telemetry.ExtractFromProtobuf(ctx, frameMsg.TraceContext)

	ctx, span := tracer.Start(ctx, "WebSocket.processFrame")
	defer span.End()

	startTime := time.Now()
	result := &pb.WebSocketFrameResponse{
		Type:    "frame_result",
		Success: false,
	}

	if frameMsg.Request == nil {
		result.Error = "missing request"
		return result
	}

	req := frameMsg.Request

	if len(req.ImageData) == 0 {
		result.Error = "missing image data"
		return result
	}

	domainImg, err := h.imageCodec.DecodeToRGB(req.ImageData)
	if err != nil {
		// Fixed: Log detailed error internally, send generic message to client
		logger.FromContext(ctx).Error().Err(err).Msg("Image decode failed")
		result.Error = "image decode failed"
		return result
	}

	opts := h.adapter.ExtractProcessingOptions(req)

	h.frameCounter++
	span.SetAttributes(
		attribute.Int("image.width", domainImg.Width),
		attribute.Int("image.height", domainImg.Height),
		attribute.String("accelerator", req.Accelerator.String()),
	)

	if h.grpcProcessor == nil {
		result.Error = "processor not available"
		return result
	}

	processedImg, err := h.grpcProcessor.ProcessImage(ctx, domainImg, opts)

	if err != nil {
		// Fixed: Log detailed error internally, send generic message to client
		// This prevents leaking internal error details that may contain sensitive information
		logger.FromContext(ctx).Error().Err(err).Msg("Frame processing failed")
		result.Error = "processing failed"
		return result
	}

	hasGrayscale := false
	for _, f := range opts.Filters {
		if f == domain.FilterGrayscale {
			hasGrayscale = true
			break
		}
	}

	encodedData, err := h.imageCodec.EncodeToPNG(processedImg, hasGrayscale)
	if err != nil {
		logger.FromContext(ctx).Error().Err(err).Msg("Image encode failed")
		result.Error = "encode failed"
		return result
	}

	result.Success = true
	result.Response = &pb.ProcessImageResponse{
		Code:      0,
		Message:   "success",
		ImageData: encodedData,
		Width:     int32(processedImg.Width),
		Height:    int32(processedImg.Height),
		Channels:  int32(len(processedImg.Data) / (processedImg.Width * processedImg.Height)),
	}

	elapsed := time.Since(startTime)
	if h.frameCounter%30 == 0 {
		log := logger.FromContext(ctx).Debug().
			Dur("elapsed", elapsed).
			Int("width", processedImg.Width).
			Int("height", processedImg.Height).
			Str("accelerator", string(opts.Accelerator))

		if len(opts.Filters) > 0 {
			log = log.Str("filter", string(opts.Filters[0]))
		}

		log.Msg("Frame processed")
	}

	return result
}

// TODO: Refactor this to use a more efficient video player
//
//nolint:gocyclo // Complex due to video frame processing pipeline
func (h *Handler) streamRealVideo(ctx context.Context, session *VideoSession, videoPath string, req *pb.StartVideoPlaybackRequest) {
	log := logger.FromContext(ctx)

	// Import video player
	player, err := video.NewFFmpegVideoPlayer(videoPath)
	if err != nil {
		log.Error().Err(err).Str("video_path", videoPath).Msg("Failed to create video player")
		return
	}

	width, height := player.GetDimensions()
	log.Info().
		Str("video_path", videoPath).
		Int("width", width).
		Int("height", height).
		Float64("fps", player.GetFPS()).
		Msg("Starting real video playback with FFmpeg")

	opts := domain.ProcessingOptions{
		Filters:       h.adapter.ToFilters(req.Filters),
		Accelerator:   h.adapter.ToAccelerator(req.Accelerator),
		GrayscaleType: h.adapter.ToGrayscaleType(req.GrayscaleType),
		BlurParams:    h.adapter.ToBlurParameters(req.BlurParams),
	}

	for _, f := range opts.Filters {
		if f == domain.FilterGrayscale {
			log.Info().Msg("Grayscale filter will be applied to video")
			break
		}
	}

	frameCallback := func(domainImg *domain.Image, frameNumber int, timestamp time.Duration) error {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		result, err := h.useCase.Execute(ctx, domainImg, opts)
		if err != nil {
			log.Error().Err(err).Int("frame", frameNumber).Msg("Failed to process video frame")
			return err
		}

		hasGrayscale := false
		for _, f := range opts.Filters {
			if f == domain.FilterGrayscale {
				hasGrayscale = true
				break
			}
		}

		encodedData, err := h.imageCodec.EncodeToPNG(result, hasGrayscale)
		if err != nil {
			log.Error().Err(err).Msg("Failed to encode video frame to PNG")
			return err
		}

		response := &pb.WebSocketFrameResponse{
			Type:    "video_frame",
			Success: true,
			Response: &pb.ProcessImageResponse{
				ImageData: encodedData,
				Width:     int32(result.Width),
				Height:    int32(result.Height),
			},
			VideoFrame: &pb.VideoFrameUpdate{
				SessionId:   session.ID,
				FrameData:   encodedData,
				FrameNumber: int32(frameNumber),
				TimestampMs: timestamp.Milliseconds(),
				IsLastFrame: false,
				FrameId:     int32(frameNumber),
			},
		}

		var responseBytes []byte
		var marshalErr error
		if session.TransportFormat == transportFormatBinary {
			responseBytes, marshalErr = proto.Marshal(response)
		} else {
			marshalOptions := protojson.MarshalOptions{
				EmitUnpopulated: true,
			}
			responseBytes, marshalErr = marshalOptions.Marshal(response)
		}

		if marshalErr != nil {
			log.Error().Err(marshalErr).Msg("Failed to marshal video frame response")
			return marshalErr
		}

		messageType := websocket.TextMessage
		if session.TransportFormat == transportFormatBinary {
			messageType = websocket.BinaryMessage
		}

		// Fixed: Use ConnID instead of Conn pointer for safe write
		if err := h.safeWriteMessage(session.ConnID, messageType, responseBytes); err != nil {
			log.Error().Err(err).Msg("Failed to send video frame")
			return err
		}

		if frameNumber%30 == 0 {
			log.Debug().Int("frame", frameNumber).Msg("Video frames streamed")
		}

		return nil
	}

	if err := player.Play(ctx, frameCallback); err != nil && !errors.Is(err, context.Canceled) {
		log.Error().Err(err).Msg("Video playback error")
	}
}

// Shutdown gracefully shuts down the handler by waiting for all active goroutines to complete
// Fixed: Prevents goroutine leaks during graceful shutdown
func (h *Handler) Shutdown(ctx context.Context) error {
	log := logger.FromContext(ctx)
	count := atomic.LoadInt64(&h.activeGoroutineCount)
	log.Info().Int64("active_goroutines", count).Msg("Shutting down WebSocket handler, waiting for active goroutines")

	done := make(chan struct{})
	go func() {
		h.activeGoroutines.Wait()
		close(done)
	}()

	select {
	case <-done:
		log.Info().Msg("All goroutines completed successfully")
		return nil
	case <-ctx.Done():
		log.Warn().Msg("Shutdown timeout, some goroutines may still be running")
		return ctx.Err()
	}
}
