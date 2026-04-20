package video

import (
	"context"
	"errors"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

var (
	ErrVideoPlaybackNotRunning = errors.New("video playback is not running for session")
)

type StopVideoPlaybackUseCaseInput struct {
	SessionID   string
	TraceContext string
	APIVersion   string
}

type StopVideoPlaybackUseCaseOutput struct {
	Code         int32
	Message      string
	SessionID    string
	Stopped      bool
	TraceContext string
	APIVersion   string
}

type StopVideoPlaybackUseCase struct {
	sessionManager *VideoSessionManager
}

func NewStopVideoPlaybackUseCase(
	sessionManager *VideoSessionManager,
) *StopVideoPlaybackUseCase {
	return &StopVideoPlaybackUseCase{
		sessionManager: sessionManager,
	}
}

func (uc *StopVideoPlaybackUseCase) Execute(
	ctx context.Context,
	input StopVideoPlaybackUseCaseInput,
) (StopVideoPlaybackUseCaseOutput, error) {
	tracer := otel.Tracer("stop-video-playback")
	ctx, span := tracer.Start(ctx, "StopVideoPlaybackUseCase.Execute",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(
		attribute.String("session_id", input.SessionID),
	)

	if input.SessionID == "" {
		span.RecordError(ErrVideoPlaybackMissingSession)
		return StopVideoPlaybackUseCaseOutput{}, ErrVideoPlaybackMissingSession
	}

	session, ok := uc.sessionManager.GetSession(input.SessionID)
	if !ok {
		span.RecordError(ErrVideoPlaybackNotRunning)
		return StopVideoPlaybackUseCaseOutput{}, ErrVideoPlaybackNotRunning
	}

	session.cancel()

	select {
	case err := <-session.done:
		if err != nil {
			span.RecordError(err)
			return StopVideoPlaybackUseCaseOutput{}, err
		}
	case <-ctx.Done():
		span.RecordError(ctx.Err())
		return StopVideoPlaybackUseCaseOutput{}, ctx.Err()
	}

	span.SetAttributes(attribute.Bool("playback.stopped", true))

	return StopVideoPlaybackUseCaseOutput{
		Code:         0,
		Message:      "Video playback stopped",
		SessionID:    input.SessionID,
		Stopped:      true,
		TraceContext: input.TraceContext,
		APIVersion:   input.APIVersion,
	}, nil
}
