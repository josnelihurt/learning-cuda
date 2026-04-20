package app

import (
	"context"

	pb "github.com/jrb/cuda-learning/proto/gen"
	systemapp "github.com/jrb/cuda-learning/src/go_api/pkg/application/platform/system"
)

type useCase[Input any, Output any] interface {
	Execute(ctx context.Context, input Input) (Output, error)
}

type processorCapabilitiesProvider interface {
	Execute(ctx context.Context, useGRPC bool) (*pb.LibraryCapabilities, systemapp.ProcessorBackendOrigin, error)
}
