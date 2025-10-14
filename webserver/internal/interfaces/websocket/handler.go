package websocket

import (
	"context"
	"log"
	"net/http"
	"time"

	"github.com/gorilla/websocket"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/webserver/internal/application"
	"github.com/jrb/cuda-learning/webserver/internal/config"
	"github.com/jrb/cuda-learning/webserver/internal/domain"
	imageinfra "github.com/jrb/cuda-learning/webserver/internal/infrastructure/image"
	"github.com/jrb/cuda-learning/webserver/internal/interfaces/adapters"
	"github.com/jrb/cuda-learning/webserver/internal/telemetry"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

type Handler struct {
	useCase      *application.ProcessImageUseCase
	imageCodec   *imageinfra.ImageCodec
	adapter      *adapters.ProtobufAdapter
	streamConfig config.StreamConfig
	frameCounter int
}

func NewHandler(useCase *application.ProcessImageUseCase, streamCfg config.StreamConfig) *Handler {
	return &Handler{
		useCase:      useCase,
		imageCodec:   imageinfra.NewImageCodec(),
		adapter:      adapters.NewProtobufAdapter(),
		streamConfig: streamCfg,
		frameCounter: 0,
	}
}

func (h *Handler) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
	ctx := telemetry.ExtractFromHTTPHeaders(r.Context(), r.Header)
	tracer := otel.Tracer("websocket-handler")

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("ws upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	transportFormat := h.streamConfig.TransportFormat
	log.Printf("WebSocket connected, transport format: %s", transportFormat)

	for {
		messageType, message, err := conn.ReadMessage()
		if err != nil {
			break
		}

		var frameMsg pb.WebSocketFrameRequest
		if transportFormat == "binary" {
			err = proto.Unmarshal(message, &frameMsg)
		} else {
			err = protojson.Unmarshal(message, &frameMsg)
		}

		if err != nil {
			log.Printf("failed to unmarshal message: %v", err)
			continue
		}

		result := h.processFrame(ctx, tracer, &frameMsg)

		var responseBytes []byte
		if transportFormat == "binary" {
			responseBytes, err = proto.Marshal(result)
		} else {
			responseBytes, err = protojson.Marshal(result)
		}

		if err != nil {
			log.Printf("failed to marshal response: %v", err)
			continue
		}

		if err := conn.WriteMessage(messageType, responseBytes); err != nil {
			break
		}
	}
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

	domainImg, err := h.imageCodec.DecodeToRGBA(req.ImageData)
	if err != nil {
		result.Error = "image decode failed"
		return result
	}

	filters := h.adapter.ToFilters(req.Filters)
	accelerator := h.adapter.ToAccelerator(req.Accelerator)
	grayscaleType := h.adapter.ToGrayscaleType(req.GrayscaleType)

	h.frameCounter++
	span.SetAttributes(
		attribute.Int("image.width", domainImg.Width),
		attribute.Int("image.height", domainImg.Height),
		attribute.String("accelerator", req.Accelerator.String()),
	)

	processedImg, err := h.useCase.Execute(ctx, domainImg, filters, accelerator, grayscaleType)
	if err != nil {
		result.Error = "processing failed: " + err.Error()
		return result
	}

	hasGrayscale := false
	for _, f := range filters {
		if f == domain.FilterGrayscale {
			hasGrayscale = true
			break
		}
	}

	encodedData, err := h.imageCodec.EncodeToPNG(processedImg, hasGrayscale)
	if err != nil {
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
		log.Printf("frame %v (%dx%d %v %v)",
			elapsed, processedImg.Width, processedImg.Height, filters, accelerator)
	}

	return result
}

