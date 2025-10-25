package application

import (
	"context"
	"fmt"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"github.com/jrb/cuda-learning/webserver/pkg/domain/interfaces"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type GetSystemInfoUseCase struct {
	processorRepo interfaces.ProcessorRepository
	configRepo    interfaces.ConfigRepository
	buildInfoRepo interfaces.BuildInfoRepository
}

func NewGetSystemInfoUseCase(
	processorRepo interfaces.ProcessorRepository,
	configRepo interfaces.ConfigRepository,
	buildInfoRepo interfaces.BuildInfoRepository,
) *GetSystemInfoUseCase {
	return &GetSystemInfoUseCase{
		processorRepo: processorRepo,
		configRepo:    configRepo,
		buildInfoRepo: buildInfoRepo,
	}
}

func (uc *GetSystemInfoUseCase) Execute(ctx context.Context) (*domain.SystemInfo, error) {
	tracer := otel.Tracer("get-system-info")
	_, span := tracer.Start(ctx, "GetSystemInfo",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	// Get processor information
	availableLibraries := uc.processorRepo.GetAvailableLibraries()
	currentLibrary := uc.processorRepo.GetCurrentLibrary()
	apiVersion := uc.processorRepo.GetAPIVersion()

	// Get environment
	environment := uc.configRepo.GetEnvironment()

	// Build system info
	systemInfo := &domain.SystemInfo{
		Version: domain.SystemVersion{
			CppVersion: apiVersion, // C++ library version
			GoVersion:  apiVersion, // Go API version (same as C++ for now)
			JsVersion:  uc.buildInfoRepo.GetVersion(),
			Branch:     uc.buildInfoRepo.GetBranch(),
			BuildTime:  uc.buildInfoRepo.GetBuildTime(),
			CommitHash: uc.buildInfoRepo.GetCommitHash(),
		},
		Environment:        environment,
		CurrentLibrary:     currentLibrary,
		APIVersion:         apiVersion,
		AvailableLibraries: availableLibraries,
	}

	// Set span attributes
	span.SetAttributes(
		attribute.String("version.cpp", systemInfo.Version.CppVersion),
		attribute.String("version.go", systemInfo.Version.GoVersion),
		attribute.String("version.js", systemInfo.Version.JsVersion),
		attribute.String("version.branch", systemInfo.Version.Branch),
		attribute.String("version.build_time", systemInfo.Version.BuildTime),
		attribute.String("version.commit_hash", systemInfo.Version.CommitHash),
		attribute.String("environment", systemInfo.Environment),
		attribute.String("processor.current_library", systemInfo.CurrentLibrary),
		attribute.String("processor.api_version", systemInfo.APIVersion),
		attribute.Int("processor.available_libraries_count", len(systemInfo.AvailableLibraries)),
	)

	if len(availableLibraries) == 0 {
		err := fmt.Errorf("no available processor libraries found")
		span.RecordError(err)
		// Don't return error, just log it
	}

	return systemInfo, nil
}
