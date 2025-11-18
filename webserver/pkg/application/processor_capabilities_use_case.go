package application

import (
	"context"
	"fmt"

	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/webserver/pkg/domain/interfaces"
)

type ProcessorBackendOrigin string

const (
	ProcessorBackendOriginCGO        ProcessorBackendOrigin = "cgo"
	ProcessorBackendOriginGRPCServer ProcessorBackendOrigin = "grpc_server"
)

type ProcessorCapabilitiesUseCase interface {
	Execute(ctx context.Context, useGRPC bool) (*pb.LibraryCapabilities, ProcessorBackendOrigin, error)
}

type processorCapabilitiesUseCase struct {
	cppRepo  interfaces.ProcessorCapabilitiesRepository
	grpcRepo interfaces.ProcessorCapabilitiesRepository
}

func NewProcessorCapabilitiesUseCase(
	cppRepo interfaces.ProcessorCapabilitiesRepository,
	grpcRepo interfaces.ProcessorCapabilitiesRepository,
) ProcessorCapabilitiesUseCase {
	return &processorCapabilitiesUseCase{
		cppRepo:  cppRepo,
		grpcRepo: grpcRepo,
	}
}

func (uc *processorCapabilitiesUseCase) Execute(ctx context.Context, useGRPC bool) (*pb.LibraryCapabilities, ProcessorBackendOrigin, error) {
	if useGRPC && uc.grpcRepo != nil {
		caps, err := uc.grpcRepo.GetCapabilities(ctx)
		if err == nil {
			return caps, ProcessorBackendOriginGRPCServer, nil
		}
	}

	if uc.cppRepo == nil {
		return nil, "", fmt.Errorf("no processor capabilities repository configured")
	}

	caps, err := uc.cppRepo.GetCapabilities(ctx)
	if err != nil {
		return nil, "", err
	}

	return caps, ProcessorBackendOriginCGO, nil
}
