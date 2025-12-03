package container

import (
	"context"
	"fmt"
	"time"

	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"github.com/jrb/cuda-learning/webserver/pkg/config"
	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/build"
	configrepo "github.com/jrb/cuda-learning/webserver/pkg/infrastructure/config"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/featureflags"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/filesystem"
	httpinfra "github.com/jrb/cuda-learning/webserver/pkg/infrastructure/http"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/logger"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/processor"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/version"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/video"
	flipt "go.flipt.io/flipt-client"
)

type Container struct {
	Config     *config.Manager
	HTTPClient *httpinfra.ClientProxy

	FeatureFlagRepo domain.FeatureFlagRepository
	FliptAPI        *featureflags.FliptHTTPAPI
	VideoRepository domain.VideoRepository

	ProcessImageUseCase        *application.ProcessImageUseCase
	EvaluateFeatureFlagUseCase *application.EvaluateFeatureFlagUseCase
	SyncFeatureFlagsUseCase    *application.SyncFeatureFlagsUseCase
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

	fliptClientProxy featureflags.FliptClientInterface
}

func New(ctx context.Context, configFile string) (*Container, error) {
	cfg := config.New(configFile)

	log := logger.New(logger.Config{
		Level:         cfg.Logging.Level,
		Format:        cfg.Logging.Format,
		Output:        cfg.Logging.Output,
		FilePath:      cfg.Logging.FilePath,
		IncludeCaller: cfg.Logging.IncludeCaller,
	})

	httpClient := httpinfra.NewInstrumentedClient(httpinfra.ClientConfig{
		Timeout:         cfg.HTTPClientTimeout,
		MaxIdleConns:    100,
		IdleConnTimeout: 90 * time.Second,
	})

	var fliptClientProxy featureflags.FliptClientInterface
	var featureFlagRepo domain.FeatureFlagRepository
	var fliptAPI *featureflags.FliptHTTPAPI

	if cfg.Flipt.Enabled {
		fliptClient, err := flipt.NewClient(
			ctx,
			flipt.WithURL(cfg.Flipt.URL),
			flipt.WithNamespace(cfg.Flipt.Namespace),
		)
		if err != nil {
			log.Warn().
				Err(err).
				Str("flipt_url", cfg.Flipt.URL).
				Msg("Failed to initialize Flipt client. Feature flags will be disabled")
			fliptClientProxy = nil
			featureFlagRepo = nil
			fliptAPI = nil
		} else {
			fliptClientProxy = featureflags.NewFliptClient(fliptClient)
			fliptWriter := featureflags.NewFliptWriter(
				cfg.Flipt.URL,
				cfg.Flipt.Namespace,
				httpClient,
			)
			featureFlagRepo = featureflags.NewFliptRepository(fliptClientProxy, fliptWriter)
			fliptAPI = featureflags.NewFliptHTTPAPI(cfg.Flipt.URL, cfg.Flipt.Namespace, httpClient)
		}
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
	syncFFUseCase := application.NewSyncFeatureFlagsUseCase(featureFlagRepo)
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

	return &Container{
		Config:                     cfg,
		HTTPClient:                 httpClient,
		FeatureFlagRepo:            featureFlagRepo,
		FliptAPI:                   fliptAPI,
		VideoRepository:            videoRepo,
		EvaluateFeatureFlagUseCase: evaluateFFUseCase,
		SyncFeatureFlagsUseCase:    syncFFUseCase,
		GetStreamConfigUseCase:     getStreamConfigUseCase,
		GetSystemInfoUseCase:       getSystemInfoUseCase,
		ListInputsUseCase:          listInputsUseCase,
		ListAvailableImagesUseCase: listAvailableImagesUseCase,
		UploadImageUseCase:         uploadImageUseCase,
		ListVideosUseCase:          listVideosUseCase,
		UploadVideoUseCase:         uploadVideoUseCase,
		GRPCProcessorClient:        grpcClient,
		fliptClientProxy:           fliptClientProxy,
	}, nil
}

func (c *Container) Close(ctx context.Context) error {
	log := logger.Global()

	if c.GRPCProcessorClient != nil {
		if err := c.GRPCProcessorClient.Close(); err != nil {
			log.Error().Err(err).Msg("Error closing gRPC processor client")
		}
	}

	if c.fliptClientProxy != nil {
		closeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		if err := c.fliptClientProxy.Close(closeCtx); err != nil {
			log.Error().Err(err).Msg("Error closing Flipt client")
			return err
		}
	}
	return nil
}
