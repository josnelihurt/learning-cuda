package video

import (
	"context"
	"errors"
	"fmt"
	"time"

	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/src/go_api/pkg/domain"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/protobuf/proto"
)

var (
	ErrVideoPlaybackAlreadyRunning = errors.New("video playback already running for session")
	ErrVideoPlaybackMissingVideoID = errors.New("video_id is required")
	ErrVideoPlaybackMissingSession = errors.New("session_id is required")
)

type StartVideoPlaybackUseCaseInput struct {
	VideoID         string
	SessionID       string
	Filters         []domain.FilterType
	Accelerator     domain.AcceleratorType
	GrayscaleType   domain.GrayscaleType
	BlurParams      *domain.BlurParameters
	GenericFilters  []*pb.GenericFilterSelection
	ModelParams     *pb.ModelInferenceParameters
	TraceContext    string
	APIVersion      string
}

type StartVideoPlaybackUseCaseOutput struct {
	Code         int32
	Message      string
	SessionID    string
	TraceContext string
	APIVersion   string
}

type StartVideoPlaybackUseCase struct {
	baseCtx         context.Context
	sessionManager  *VideoSessionManager
	videoRepository videoRepository
	playerFactory   StreamVideoPlayerFactory
	peerFactory     StreamVideoPeerFactory
}

func NewStartVideoPlaybackUseCase(
	baseCtx context.Context,
	sessionManager *VideoSessionManager,
	videoRepository videoRepository,
	playerFactory StreamVideoPlayerFactory,
	peerFactory StreamVideoPeerFactory,
) *StartVideoPlaybackUseCase {
	if baseCtx == nil {
		baseCtx = context.Background()
	}

	return &StartVideoPlaybackUseCase{
		baseCtx:         baseCtx,
		sessionManager:  sessionManager,
		videoRepository: videoRepository,
		playerFactory:   playerFactory,
		peerFactory:     peerFactory,
	}
}

func (uc *StartVideoPlaybackUseCase) Execute(
	ctx context.Context,
	input StartVideoPlaybackUseCaseInput,
) (StartVideoPlaybackUseCaseOutput, error) {
	tracer := otel.Tracer("start-video-playback")
	ctx, span := tracer.Start(ctx, "StartVideoPlaybackUseCase.Execute",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(
		attribute.String("video_id", input.VideoID),
		attribute.String("session_id", input.SessionID),
	)

	if input.VideoID == "" {
		span.RecordError(ErrVideoPlaybackMissingVideoID)
		return StartVideoPlaybackUseCaseOutput{}, ErrVideoPlaybackMissingVideoID
	}
	if input.SessionID == "" {
		span.RecordError(ErrVideoPlaybackMissingSession)
		return StartVideoPlaybackUseCaseOutput{}, ErrVideoPlaybackMissingSession
	}

	if uc.sessionManager.SessionExists(input.SessionID) {
		span.RecordError(ErrVideoPlaybackAlreadyRunning)
		return StartVideoPlaybackUseCaseOutput{}, ErrVideoPlaybackAlreadyRunning
	}

	video, err := uc.videoRepository.GetByID(ctx, input.VideoID)
	if err != nil {
		span.RecordError(err)
		return StartVideoPlaybackUseCaseOutput{}, fmt.Errorf("get video by id: %w", err)
	}
	span.SetAttributes(attribute.String("video.path", video.Path))

	player, err := uc.playerFactory(video.Path)
	if err != nil {
		span.RecordError(err)
		return StartVideoPlaybackUseCaseOutput{}, fmt.Errorf("create video player: %w", err)
	}

	peer, err := uc.peerFactory(input.SessionID)
	if err != nil {
		span.RecordError(err)
		return StartVideoPlaybackUseCaseOutput{}, fmt.Errorf("create webrtc peer: %w", err)
	}

	if err := peer.Connect(ctx); err != nil {
		span.RecordError(err)
		return StartVideoPlaybackUseCaseOutput{}, fmt.Errorf("connect webrtc peer %q: %w", peer.Label(), err)
	}
	span.SetAttributes(attribute.String("peer.label", peer.Label()))

	playbackCtx, cancel := context.WithCancel(uc.baseCtx)
	session := &videoPlaybackSession{
		cancel: cancel,
		done:   make(chan error, 1),
		peer:   peer,
	}

	if err := uc.sessionManager.CreateSession(input.SessionID, session); err != nil {
		cancel()
		_ = peer.Close()
		if errors.Is(err, ErrSessionAlreadyExists) {
			span.RecordError(ErrVideoPlaybackAlreadyRunning)
			return StartVideoPlaybackUseCaseOutput{}, ErrVideoPlaybackAlreadyRunning
		}
		span.RecordError(err)
		return StartVideoPlaybackUseCaseOutput{}, fmt.Errorf("create session: %w", err)
	}

	go uc.runPlayback(playbackCtx, input.SessionID, input, player, session)

	span.SetAttributes(attribute.Bool("playback.started", true))

	return StartVideoPlaybackUseCaseOutput{
		Code:         0,
		Message:      "Video playback started",
		SessionID:    input.SessionID,
		TraceContext: input.TraceContext,
		APIVersion:   input.APIVersion,
	}, nil
}

func (uc *StartVideoPlaybackUseCase) runPlayback(
	ctx context.Context,
	sessionID string,
	input StartVideoPlaybackUseCaseInput,
	player StreamVideoPlayer,
	session *videoPlaybackSession,
) {
	err := player.Play(ctx, func(frame *domain.Image, frameNumber int, _ time.Duration) error {
		payload, marshalErr := proto.Marshal(buildVideoFrameRequest(input, frame))
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

	uc.sessionManager.DeleteSession(sessionID)

	session.done <- err
	close(session.done)
}

func buildVideoFrameRequest(input StartVideoPlaybackUseCaseInput, frame *domain.Image) *pb.ProcessImageRequest {
	return &pb.ProcessImageRequest{
		ImageData:      frame.Data,
		Width:          int32(frame.Width),
		Height:         int32(frame.Height),
		Channels:       inferChannels(frame),
		Filters:        convertFilters(input.Filters),
		Accelerator:    convertAccelerator(input.Accelerator),
		GrayscaleType:  convertGrayscaleType(input.GrayscaleType),
		BlurParams:     convertBlurParams(input.BlurParams),
		GenericFilters: input.GenericFilters,
		ModelParams:    input.ModelParams,
		SessionId:      input.SessionID,
		TraceContext:   &pb.TraceContext{Traceparent: input.TraceContext},
		ApiVersion:     input.APIVersion,
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

func convertFilters(filters []domain.FilterType) []pb.FilterType {
	if filters == nil {
		return nil
	}
	result := make([]pb.FilterType, 0, len(filters))
	for _, f := range filters {
		switch f {
		case domain.FilterNone:
			result = append(result, pb.FilterType_FILTER_TYPE_NONE)
		case domain.FilterGrayscale:
			result = append(result, pb.FilterType_FILTER_TYPE_GRAYSCALE)
		case domain.FilterBlur:
			result = append(result, pb.FilterType_FILTER_TYPE_BLUR)
		case domain.FilterModel:
			result = append(result, pb.FilterType_FILTER_TYPE_MODEL_INFERENCE)
		default:
			result = append(result, pb.FilterType_FILTER_TYPE_NONE)
		}
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

func convertAccelerator(acc domain.AcceleratorType) pb.AcceleratorType {
	switch acc {
	case domain.AcceleratorGPU:
		return pb.AcceleratorType_ACCELERATOR_TYPE_CUDA
	case domain.AcceleratorCPU:
		return pb.AcceleratorType_ACCELERATOR_TYPE_CPU
	default:
		return pb.AcceleratorType_ACCELERATOR_TYPE_CPU
	}
}

func convertGrayscaleType(gt domain.GrayscaleType) pb.GrayscaleType {
	switch gt {
	case domain.GrayscaleBT601:
		return pb.GrayscaleType_GRAYSCALE_TYPE_BT601
	case domain.GrayscaleBT709:
		return pb.GrayscaleType_GRAYSCALE_TYPE_BT709
	case domain.GrayscaleAverage:
		return pb.GrayscaleType_GRAYSCALE_TYPE_AVERAGE
	case domain.GrayscaleLightness:
		return pb.GrayscaleType_GRAYSCALE_TYPE_LIGHTNESS
	case domain.GrayscaleLuminosity:
		return pb.GrayscaleType_GRAYSCALE_TYPE_LUMINOSITY
	default:
		return pb.GrayscaleType_GRAYSCALE_TYPE_BT601
	}
}

func convertBlurParams(bp *domain.BlurParameters) *pb.GaussianBlurParameters {
	if bp == nil {
		return nil
	}
	var borderMode pb.BorderMode
	switch bp.BorderMode {
	case domain.BorderModeClamp:
		borderMode = pb.BorderMode_BORDER_MODE_CLAMP
	case domain.BorderModeReflect:
		borderMode = pb.BorderMode_BORDER_MODE_REFLECT
	case domain.BorderModeWrap:
		borderMode = pb.BorderMode_BORDER_MODE_WRAP
	default:
		borderMode = pb.BorderMode_BORDER_MODE_REFLECT
	}
	return &pb.GaussianBlurParameters{
		KernelSize: bp.KernelSize,
		Sigma:      bp.Sigma,
		BorderMode: borderMode,
		Separable:  bp.Separable,
	}
}
