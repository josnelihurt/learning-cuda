package application

import (
	"context"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"github.com/jrb/cuda-learning/webserver/pkg/domain/interfaces"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type GetSystemInfoUseCase struct {
	configRepo    interfaces.ConfigRepository
	buildInfoRepo interfaces.BuildInfoRepository
	versionRepo   interfaces.VersionRepository
}

func NewGetSystemInfoUseCase(
	configRepo interfaces.ConfigRepository,
	buildInfoRepo interfaces.BuildInfoRepository,
	versionRepo interfaces.VersionRepository,
) *GetSystemInfoUseCase {
	return &GetSystemInfoUseCase{
		configRepo:    configRepo,
		buildInfoRepo: buildInfoRepo,
		versionRepo:   versionRepo,
	}
}

func (uc *GetSystemInfoUseCase) Execute(ctx context.Context) (*domain.SystemInfo, error) {
	tracer := otel.Tracer("get-system-info")
	_, span := tracer.Start(ctx, "GetSystemInfo",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	// Get environment
	environment := uc.configRepo.GetEnvironment()

	// Build system info with versions from VERSION files and shared build info
	systemInfo := &domain.SystemInfo{
		Version: domain.SystemVersion{
			GoVersion:    uc.versionRepo.GetGoVersion(),
			CppVersion:   uc.versionRepo.GetCppVersion(),
			ProtoVersion: uc.versionRepo.GetProtoVersion(),
			Branch:       uc.buildInfoRepo.GetBranch(),
			BuildTime:    uc.buildInfoRepo.GetBuildTime(),
			CommitHash:   uc.buildInfoRepo.GetCommitHash(),
		},
		Environment: environment,
	}

	// Set span attributes
	span.SetAttributes(
		attribute.String("version.go", systemInfo.Version.GoVersion),
		attribute.String("version.cpp", systemInfo.Version.CppVersion),
		attribute.String("version.proto", systemInfo.Version.ProtoVersion),
		attribute.String("version.branch", systemInfo.Version.Branch),
		attribute.String("version.build_time", systemInfo.Version.BuildTime),
		attribute.String("version.commit_hash", systemInfo.Version.CommitHash),
		attribute.String("environment", systemInfo.Environment),
	)

	return systemInfo, nil
}
