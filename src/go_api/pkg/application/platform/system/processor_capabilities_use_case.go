package system

import (
	"context"
	"fmt"

	pb "github.com/jrb/cuda-learning/proto/gen"
)

type ProcessorBackendOrigin string

const (
	ProcessorBackendOriginGRPCServer ProcessorBackendOrigin = "grpc_server"
)

type ProcessorCapabilitiesUseCase struct {
	grpcRepo processorCapabilitiesRepository
}

func NewProcessorCapabilitiesUseCase(
	grpcRepo processorCapabilitiesRepository,
) *ProcessorCapabilitiesUseCase {
	return &ProcessorCapabilitiesUseCase{
		grpcRepo: grpcRepo,
	}
}

func (uc *ProcessorCapabilitiesUseCase) Execute(ctx context.Context, useGRPC bool) (*pb.LibraryCapabilities, ProcessorBackendOrigin, error) {
	if uc.grpcRepo == nil {
		return nil, "", fmt.Errorf("gRPC processor capabilities repository required")
	}

	caps, err := uc.grpcRepo.GetCapabilities(ctx)
	if err != nil {
		return nil, "", fmt.Errorf("failed to get capabilities via gRPC: %w", err)
	}

	return caps, ProcessorBackendOriginGRPCServer, nil
}
