package application

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/src/go_api/pkg/domain"
	"google.golang.org/protobuf/proto"
)

var (
	ErrVideoPlaybackAlreadyRunning = errors.New("video playback already running for session")
	ErrVideoPlaybackNotRunning     = errors.New("video playback is not running for session")
	ErrVideoPlaybackMissingVideoID = errors.New("video_id is required")
	ErrVideoPlaybackMissingSession = errors.New("session_id is required")
)

type StreamVideoPlayer interface {
	Play(ctx context.Context, frameCallback func(*domain.Image, int, time.Duration) error) error
}

type StreamVideoPlayerFactory func(videoPath string) (StreamVideoPlayer, error)

type StreamVideoPeer interface {
	Connect(ctx context.Context) error
	Send(payload []byte) error
	Close() error
	Label() string
}

type StreamVideoPeerFactory func(browserSessionID string) (StreamVideoPeer, error)

type StreamVideoUseCase struct {
	baseCtx         context.Context
	videoRepository domain.VideoRepository
	playerFactory   StreamVideoPlayerFactory
	peerFactory     StreamVideoPeerFactory

	mu       sync.Mutex
	sessions map[string]*videoPlaybackSession
}

type videoPlaybackSession struct {
	cancel context.CancelFunc
	done   chan error
	peer   StreamVideoPeer
}

func NewStreamVideoUseCase(
	baseCtx context.Context,
	videoRepository domain.VideoRepository,
	playerFactory StreamVideoPlayerFactory,
	peerFactory StreamVideoPeerFactory,
) *StreamVideoUseCase {
	if baseCtx == nil {
		baseCtx = context.Background()
	}

	return &StreamVideoUseCase{
		baseCtx:         baseCtx,
		videoRepository: videoRepository,
		playerFactory:   playerFactory,
		peerFactory:     peerFactory,
		sessions:        make(map[string]*videoPlaybackSession),
	}
}

func (uc *StreamVideoUseCase) Start(
	ctx context.Context,
	req *pb.StartVideoPlaybackRequest,
) (*pb.StartVideoPlaybackResponse, error) {
	if req.GetVideoId() == "" {
		return nil, ErrVideoPlaybackMissingVideoID
	}
	if req.GetSessionId() == "" {
		return nil, ErrVideoPlaybackMissingSession
	}
	if uc.isSessionActive(req.GetSessionId()) {
		return nil, ErrVideoPlaybackAlreadyRunning
	}

	video, err := uc.videoRepository.GetByID(ctx, req.GetVideoId())
	if err != nil {
		return nil, fmt.Errorf("get video by id: %w", err)
	}

	player, err := uc.playerFactory(video.Path)
	if err != nil {
		return nil, fmt.Errorf("create video player: %w", err)
	}

	peer, err := uc.peerFactory(req.GetSessionId())
	if err != nil {
		return nil, fmt.Errorf("create webrtc peer: %w", err)
	}

	if err := peer.Connect(ctx); err != nil {
		return nil, fmt.Errorf("connect webrtc peer %q: %w", peer.Label(), err)
	}

	playbackCtx, cancel := context.WithCancel(uc.baseCtx)
	session := &videoPlaybackSession{
		cancel: cancel,
		done:   make(chan error, 1),
		peer:   peer,
	}

	uc.mu.Lock()
	if _, exists := uc.sessions[req.GetSessionId()]; exists {
		uc.mu.Unlock()
		cancel()
		_ = peer.Close()
		return nil, ErrVideoPlaybackAlreadyRunning
	}
	uc.sessions[req.GetSessionId()] = session
	uc.mu.Unlock()

	go uc.runPlayback(playbackCtx, req.GetSessionId(), req, player, session)

	return &pb.StartVideoPlaybackResponse{
		Code:         0,
		Message:      "Video playback started",
		SessionId:    req.GetSessionId(),
		TraceContext: req.GetTraceContext(),
		ApiVersion:   req.GetApiVersion(),
	}, nil
}

func (uc *StreamVideoUseCase) Stop(
	ctx context.Context,
	req *pb.StopVideoPlaybackRequest,
) (*pb.StopVideoPlaybackResponse, error) {
	if req.GetSessionId() == "" {
		return nil, ErrVideoPlaybackMissingSession
	}

	session, ok := uc.getSession(req.GetSessionId())
	if !ok {
		return nil, ErrVideoPlaybackNotRunning
	}

	session.cancel()

	select {
	case err := <-session.done:
		if err != nil {
			return nil, err
		}
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	return &pb.StopVideoPlaybackResponse{
		Code:         0,
		Message:      "Video playback stopped",
		SessionId:    req.GetSessionId(),
		Stopped:      true,
		TraceContext: req.GetTraceContext(),
		ApiVersion:   req.GetApiVersion(),
	}, nil
}

func (uc *StreamVideoUseCase) runPlayback(
	ctx context.Context,
	sessionID string,
	req *pb.StartVideoPlaybackRequest,
	player StreamVideoPlayer,
	session *videoPlaybackSession,
) {
	err := player.Play(ctx, func(frame *domain.Image, frameNumber int, _ time.Duration) error {
		payload, marshalErr := proto.Marshal(buildVideoFrameRequest(req, frame))
		if marshalErr != nil {
			return fmt.Errorf("marshal frame %d: %w", frameNumber, marshalErr)
		}

		if sendErr := session.peer.Send(payload); sendErr != nil {
			return fmt.Errorf("send frame %d: %w", frameNumber, sendErr)
		}

		return nil
	})

	if closeErr := session.peer.Close(); closeErr != nil && err == nil {
		err = closeErr
	}

	if errors.Is(err, context.Canceled) {
		err = nil
	}

	uc.mu.Lock()
	delete(uc.sessions, sessionID)
	uc.mu.Unlock()

	session.done <- err
	close(session.done)
}

func buildVideoFrameRequest(req *pb.StartVideoPlaybackRequest, frame *domain.Image) *pb.ProcessImageRequest {
	return &pb.ProcessImageRequest{
		ImageData:      frame.Data,
		Width:          int32(frame.Width),
		Height:         int32(frame.Height),
		Channels:       inferChannels(frame),
		Filters:        req.GetFilters(),
		Accelerator:    req.GetAccelerator(),
		GrayscaleType:  req.GetGrayscaleType(),
		BlurParams:     req.GetBlurParams(),
		GenericFilters: req.GetGenericFilters(),
		SessionId:      req.GetSessionId(),
		TraceContext:   req.GetTraceContext(),
		ApiVersion:     req.GetApiVersion(),
	}
}

func inferChannels(frame *domain.Image) int32 {
	if frame == nil || frame.Width <= 0 || frame.Height <= 0 {
		return 3
	}

	pixelCount := frame.Width * frame.Height
	if pixelCount <= 0 {
		return 3
	}

	channels := len(frame.Data) / pixelCount
	if channels <= 0 {
		return 3
	}

	return int32(channels)
}

func (uc *StreamVideoUseCase) isSessionActive(sessionID string) bool {
	uc.mu.Lock()
	defer uc.mu.Unlock()

	_, exists := uc.sessions[sessionID]
	return exists
}

func (uc *StreamVideoUseCase) getSession(sessionID string) (*videoPlaybackSession, bool) {
	uc.mu.Lock()
	defer uc.mu.Unlock()

	session, exists := uc.sessions[sessionID]
	return session, exists
}
