package container

import (
	"context"
	"fmt"
	"time"

	"github.com/jrb/cuda-learning/src/go_api/pkg/application"
	"github.com/jrb/cuda-learning/src/go_api/pkg/config"
	"github.com/jrb/cuda-learning/src/go_api/pkg/domain"
	domainInterfaces "github.com/jrb/cuda-learning/src/go_api/pkg/domain/interfaces"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/build"
	configrepo "github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/config"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/featureflags"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/filesystem"
	httpinfra "github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/http"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/logger"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/mqtt"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/processor"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/version"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/video"
)

type Container struct {
	Config     *config.Manager
	HTTPClient *httpinfra.ClientProxy

	FeatureFlagRepo domain.FeatureFlagRepository
	VideoRepository domain.VideoRepository

	ProcessImageUseCase        *application.ProcessImageUseCase
	EvaluateFeatureFlagUseCase *application.EvaluateFeatureFlagUseCase
	GetStreamConfigUseCase     *application.GetStreamConfigUseCase
	GetSystemInfoUseCase       *application.GetSystemInfoUseCase
	ListInputsUseCase          *application.ListInputsUseCase
	// ListAvailableImagesUseCase handles listing available images
	// language: english-only
	ListAvailableImagesUseCase *application.ListAvailableImagesUseCase //nolint:language
	UploadImageUseCase         *application.UploadImageUseCase
	ListVideosUseCase          *application.ListVideosUseCase
	UploadVideoUseCase         *application.UploadVideoUseCase

	GRPCProcessorClient *processor.GRPCClient
	DeviceMonitor       domainInterfaces.MQTTDeviceMonitor
}

func New(ctx context.Context, configFile string) (*Container, error) {
	cfg := config.New(configFile)

	//nolint:contextcheck // logger.New creates its own context for OTLP initialization
	log := logger.New(&logger.Config{
		Level:             cfg.Logging.Level,
		Format:            cfg.Logging.Format,
		Output:            cfg.Logging.Output,
		FilePath:          cfg.Logging.FilePath,
		IncludeCaller:     cfg.Logging.IncludeCaller,
		RemoteEnabled:     cfg.Logging.RemoteEnabled,
		RemoteEndpoint:    cfg.Observability.OtelCollectorHTTPEndpoint,
		RemoteEnvironment: cfg.Logging.RemoteEnvironment,
		ServiceName:       cfg.Observability.ServiceName,
	})
	log.Info().Str("config_file", configFile).Any("config", cfg).Msg("Container initialized")

	httpClient := httpinfra.NewInstrumentedClient(httpinfra.ClientConfig{
		Timeout:         cfg.HTTPClientTimeout,
		MaxIdleConns:    100,
		IdleConnTimeout: 90 * time.Second,
	})

	var featureFlagRepo domain.FeatureFlagRepository

	if cfg.GoFeatureFlag.Enabled {
		goffRepo := featureflags.NewGoffRepository(cfg.GoFeatureFlag.FilePath)
		if err := goffRepo.ValidateConfig(); err != nil {
			return nil, fmt.Errorf("invalid go feature flag config: %w", err)
		}
		featureFlagRepo = goffRepo
	}

	buildInfo := build.NewBuildInfo()

	// Create repository implementations
	configRepo := configrepo.NewConfigRepository(cfg)
	buildInfoRepo := build.NewBuildInfoRepository(buildInfo)
	versionRepo := version.NewVersionRepository()

	if cfg.Processor.GRPCServerAddress == "" {
		return nil, fmt.Errorf("processor.grpc_server_address is required but not configured")
	}

	grpcClient, err := processor.NewGRPCClient(ctx, processor.GRPCClientConfig{
		Address:      cfg.Processor.GRPCServerAddress,
		DialTimeout:  5 * time.Second,
		MaxRecvBytes: 64 * 1024 * 1024,
		MaxSendBytes: 64 * 1024 * 1024,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to initialize gRPC processor client (required): %w", err)
	}
	log.Info().
		Str("grpc_address", cfg.Processor.GRPCServerAddress).
		Msg("gRPC client initialized successfully")

	evaluateFFUseCase := application.NewEvaluateFeatureFlagUseCase(featureFlagRepo)
	getStreamConfigUseCase := application.NewGetStreamConfigUseCase(evaluateFFUseCase, cfg.Stream)

	getSystemInfoUseCase := application.NewGetSystemInfoUseCase(configRepo, buildInfoRepo, versionRepo)

	videoRepo := video.NewFileVideoRepository(context.Background(), "data/videos", "data/video_previews") //nolint:contextcheck
	listInputsUseCase := application.NewListInputsUseCase(videoRepo)

	staticImageRepo := filesystem.NewStaticImageRepository(cfg.StaticImages.Directory)
	// Create use case for listing available images
	// language: english-only
	listAvailableImagesUseCase := application.NewListAvailableImagesUseCase(staticImageRepo) //nolint:language
	uploadImageUseCase := application.NewUploadImageUseCase(staticImageRepo)

	listVideosUseCase := application.NewListVideosUseCase(videoRepo)
	uploadVideoUseCase := application.NewUploadVideoUseCase(videoRepo, "data/videos", "data/video_previews")

	var deviceMonitor domainInterfaces.MQTTDeviceMonitor
	if cfg.MQTT.Broker != "" {
		monitor, err := mqtt.NewDeviceMonitor(cfg.MQTT)
		if err != nil {
			log.Warn().Err(err).Msg("Failed to initialize MQTT device monitor")
		} else {
			deviceMonitor = monitor
			log.Info().Msg("MQTT device monitor initialized")
		}
	}

	return &Container{
		Config:                     cfg,
		HTTPClient:                 httpClient,
		FeatureFlagRepo:            featureFlagRepo,
		VideoRepository:            videoRepo,
		EvaluateFeatureFlagUseCase: evaluateFFUseCase,
		GetStreamConfigUseCase:     getStreamConfigUseCase,
		GetSystemInfoUseCase:       getSystemInfoUseCase,
		ListInputsUseCase:          listInputsUseCase,
		ListAvailableImagesUseCase: listAvailableImagesUseCase,
		UploadImageUseCase:         uploadImageUseCase,
		ListVideosUseCase:          listVideosUseCase,
		UploadVideoUseCase:         uploadVideoUseCase,
		GRPCProcessorClient:        grpcClient,
		DeviceMonitor:              deviceMonitor,
	}, nil
}

func (c *Container) Close(ctx context.Context) error {
	log := logger.Global()

	if c.GRPCProcessorClient != nil {
		if err := c.GRPCProcessorClient.Close(); err != nil {
			log.Error().Err(err).Msg("Error closing gRPC processor client")
		}
	}

	return nil
}
