package container

import (
	"context"

	"sync"
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
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/processor/loader"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/video"
	flipt "go.flipt.io/flipt-client"
)

type Container struct {
	Config     *config.Manager
	HTTPClient *httpinfra.ClientProxy

	FeatureFlagRepo domain.FeatureFlagRepository
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

	CppConnector      *processor.CppConnector
	ProcessorRegistry *loader.Registry
	ProcessorLoader   *loader.Loader
	LoaderMutex       *sync.RWMutex

	fliptClientProxy featureflags.FliptClientInterface
}

func New(ctx context.Context) (*Container, error) {
	cfg := config.New()

	log := logger.New(logger.Config{
		Level:         cfg.Logging.Level,
		Format:        cfg.Logging.Format,
		Output:        cfg.Logging.Output,
		IncludeCaller: cfg.Logging.IncludeCaller,
	})

	httpClient := httpinfra.NewInstrumentedClient(httpinfra.ClientConfig{
		Timeout:         cfg.HTTPClientTimeout,
		MaxIdleConns:    100,
		IdleConnTimeout: 90 * time.Second,
	})

	var fliptClientProxy featureflags.FliptClientInterface
	var featureFlagRepo domain.FeatureFlagRepository

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
	} else {
		fliptClientProxy = featureflags.NewFliptClient(fliptClient)
		fliptWriter := featureflags.NewFliptWriter(
			cfg.Flipt.URL,
			cfg.Flipt.Namespace,
			httpClient,
		)
		featureFlagRepo = featureflags.NewFliptRepository(fliptClientProxy, fliptWriter)
	}

	registry := loader.NewRegistry(cfg.Processor.LibraryBasePath)
	if discoverErr := registry.Discover(); discoverErr != nil {
		log.Warn().
			Err(discoverErr).
			Str("library_path", cfg.Processor.LibraryBasePath).
			Msg("Failed to discover processor libraries")
	}

	libInfo, err := registry.GetByVersion(cfg.Processor.DefaultLibrary)
	if err != nil {
		return nil, err
	}

	processorLoader, err := registry.LoadLibrary(cfg.Processor.DefaultLibrary)
	if err != nil {
		return nil, err
	}

	cppConnector, err := processor.New(libInfo.Path)
	if err != nil {
		return nil, err
	}

	evaluateFFUseCase := application.NewEvaluateFeatureFlagUseCase(featureFlagRepo)
	syncFFUseCase := application.NewSyncFeatureFlagsUseCase(featureFlagRepo)
	getStreamConfigUseCase := application.NewGetStreamConfigUseCase(evaluateFFUseCase, cfg.Stream)

	buildInfo := build.NewBuildInfo()

	// Create repository implementations
	processorRepo := processor.NewProcessorRepository(registry)
	configRepo := configrepo.NewConfigRepository(cfg)
	buildInfoRepo := build.NewBuildInfoRepository(buildInfo)

	getSystemInfoUseCase := application.NewGetSystemInfoUseCase(processorRepo, configRepo, buildInfoRepo)

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
		CppConnector:               cppConnector,
		ProcessorRegistry:          registry,
		ProcessorLoader:            processorLoader,
		LoaderMutex:                &sync.RWMutex{},
		fliptClientProxy:           fliptClientProxy,
	}, nil
}

func (c *Container) Close(ctx context.Context) error {
	log := logger.Global()

	if c.CppConnector != nil {
		c.CppConnector.Close()
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
