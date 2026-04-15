package system

import (
	"context"
	"fmt"

	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/src/go_api/pkg/domain/interfaces"
)

type ProcessorBackendOrigin string

const (
	ProcessorBackendOriginCGO        ProcessorBackendOrigin = "cgo"
	ProcessorBackendOriginGRPCServer ProcessorBackendOrigin = "grpc_server"
)

type ProcessorCapabilitiesUseCase struct {
	cppRepo  interfaces.ProcessorCapabilitiesRepository
	grpcRepo interfaces.ProcessorCapabilitiesRepository
}

func NewProcessorCapabilitiesUseCase(
	cppRepo interfaces.ProcessorCapabilitiesRepository,
	grpcRepo interfaces.ProcessorCapabilitiesRepository,
) *ProcessorCapabilitiesUseCase {
	return &ProcessorCapabilitiesUseCase{
		cppRepo:  cppRepo,
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
