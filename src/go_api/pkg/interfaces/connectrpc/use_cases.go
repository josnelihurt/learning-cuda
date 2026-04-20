package connectrpc

import (
	"context"

	pb "github.com/jrb/cuda-learning/proto/gen"
	systemapp "github.com/jrb/cuda-learning/src/go_api/pkg/application/platform/system"
)

type useCase[Input any, Output any] interface {
	Execute(ctx context.Context, input Input) (Output, error)
}

type streamVideoUseCase interface {
	Start(ctx context.Context, req *pb.StartVideoPlaybackRequest) (*pb.StartVideoPlaybackResponse, error)
	Stop(ctx context.Context, req *pb.StopVideoPlaybackRequest) (*pb.StopVideoPlaybackResponse, error)
}

type processorCapabilitiesUseCase interface {
	Execute(ctx context.Context, useGRPC bool) (*pb.LibraryCapabilities, systemapp.ProcessorBackendOrigin, error)
}

type evaluateFeatureFlagUseCase interface {
	EvaluateBoolean(ctx context.Context, flagKey string, entityID string, fallbackValue bool) (bool, error)
	EvaluateString(ctx context.Context, flagKey string, entityID string, fallbackValue string) (string, error)
}
