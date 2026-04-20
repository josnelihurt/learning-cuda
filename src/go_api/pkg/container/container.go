package container

import (
	"context"
	"fmt"
	"time"

	ffapp "github.com/jrb/cuda-learning/src/go_api/pkg/application/flags"
	imageapp "github.com/jrb/cuda-learning/src/go_api/pkg/application/media/image"
	videoapp "github.com/jrb/cuda-learning/src/go_api/pkg/application/media/video"
	systemapp "github.com/jrb/cuda-learning/src/go_api/pkg/application/platform/system"
	"github.com/jrb/cuda-learning/src/go_api/pkg/config"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/build"
	configrepo "github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/config"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/featureflags"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/filesystem"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/logger"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/mqtt"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/processor"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/version"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/video"
	webrtcinfra "github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/webrtc"
)

type Container struct {
	Config *config.Manager

	FeatureFlagRepo featureFlagRepository
	VideoRepository videoRepository

	ProcessImageUseCase               useCase[imageapp.ProcessImageUseCaseInput, imageapp.ProcessImageUseCaseOutput]
	EvaluateFeatureFlagBooleanUseCase useCase[ffapp.EvaluateFeatureFlagBooleanUseCaseInput, ffapp.EvaluateFeatureFlagBooleanUseCaseOutput]
	EvaluateFeatureFlagStringUseCase  useCase[ffapp.EvaluateFeatureFlagStringUseCaseInput, ffapp.EvaluateFeatureFlagStringUseCaseOutput]
	GetSystemInfoUseCase              useCase[systemapp.GetSystemInfoUseCaseInput, systemapp.GetSystemInfoUseCaseOutput]
	ListInputsUseCase                 useCase[videoapp.ListInputsUseCaseInput, videoapp.ListInputsUseCaseOutput]
	ListAvailableImagesUseCase        useCase[imageapp.ListAvailableImagesUseCaseInput, imageapp.ListAvailableImagesUseCaseOutput]
	UploadImageUseCase                useCase[imageapp.UploadImageUseCaseInput, imageapp.UploadImageUseCaseOutput]
	ListVideosUseCase                 useCase[videoapp.ListVideosUseCaseInput, videoapp.ListVideosUseCaseOutput]
	UploadVideoUseCase                useCase[videoapp.UploadVideoUseCaseInput, videoapp.UploadVideoUseCaseOutput]
	// TODO: replace with streamVideoUseCase this is not possible right now because the StreamVideoUseCase depends on proto objects
	StreamVideoUseCase *videoapp.StreamVideoUseCase

	AcceleratorRegistry *processor.Registry
	AcceleratorControl  *processor.ControlServer
	DeviceMonitor       *mqtt.DeviceMonitor
}

func New(ctx context.Context, configFile string) (*Container, error) {
	cfg := config.New(configFile)

	log := logger.New(&logger.Config{
		Level:             cfg.Logging.Level,
		Format:            cfg.Logging.Format,
		Output:            cfg.Logging.Output,
		FilePath:          cfg.Logging.FilePath,
		IncludeCaller:     cfg.Logging.IncludeCaller,
		RemoteEnabled:     cfg.Logging.RemoteEnabled,
		RemoteEndpoint:    cfg.Logging.RemoteEndpoint,
		RemoteEnvironment: cfg.Logging.RemoteEnvironment,
		ServiceName:       cfg.Observability.ServiceName,
	})
	log.Info().Str("config_file", configFile).Any("config", cfg).Msg("Container initialized")

	var featureFlagRepo featureFlagRepository

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

	registry := processor.NewRegistry(log)

	controlServer, err := processor.NewControlServer(cfg.Processor, registry)
	if err != nil {
		err = fmt.Errorf("control server not initialized (certs missing?); accelerator connections will fail: %w", err)
		return nil, err
	}
	log.Info().
		Str("listen_address", cfg.Processor.ListenAddress).
		Msg("accelerator control server created")

	evaluateFFBooleanUseCase := ffapp.NewEvaluateFeatureFlagBooleanUseCase(featureFlagRepo)
	evaluateFFStringUseCase := ffapp.NewEvaluateFeatureFlagStringUseCase(featureFlagRepo)

	getSystemInfoUseCase := systemapp.NewGetSystemInfoUseCase(configRepo, buildInfoRepo, versionRepo)

	videoRepo := video.NewFileVideoRepository(ctx, "data/videos", "data/video_previews")
	listInputsUseCase := videoapp.NewListInputsUseCase(videoRepo)

	staticImageRepo := filesystem.NewStaticImageRepository(cfg.StaticImages.Directory)
	listAvailableImagesUseCase := imageapp.NewListAvailableImagesUseCase(staticImageRepo) //nolint:language
	uploadImageUseCase := imageapp.NewUploadImageUseCase(staticImageRepo)

	listVideosUseCase := videoapp.NewListVideosUseCase(videoRepo)
	uploadVideoUseCase := videoapp.NewUploadVideoUseCase(videoRepo, "data/videos", "data/video_previews")
	streamVideoUseCase := videoapp.NewStreamVideoUseCase(
		ctx,
		videoRepo,
		func(videoPath string) (videoapp.StreamVideoPlayer, error) {
			return video.NewFFmpegVideoPlayer(videoPath)
		},
		func(browserSessionID string) (videoapp.StreamVideoPeer, error) {
			return webrtcinfra.NewGoPeer(browserSessionID), nil
		},
	)

	var deviceMonitor *mqtt.DeviceMonitor
	if cfg.MQTT.Broker != "" {
		monitor, err := mqtt.NewDeviceMonitor(ctx, cfg.MQTT)
		if err != nil {
			log.Warn().Err(err).Msg("Failed to initialize MQTT device monitor")
		} else {
			deviceMonitor = monitor
			log.Info().Msg("MQTT device monitor initialized")
		}
	}

	return &Container{
		Config:                            cfg,
		FeatureFlagRepo:                   featureFlagRepo,
		VideoRepository:                   videoRepo,
		EvaluateFeatureFlagBooleanUseCase: evaluateFFBooleanUseCase,
		EvaluateFeatureFlagStringUseCase:  evaluateFFStringUseCase,
		GetSystemInfoUseCase:              getSystemInfoUseCase,
		ListInputsUseCase:                 listInputsUseCase,
		ListAvailableImagesUseCase:        listAvailableImagesUseCase,
		UploadImageUseCase:                uploadImageUseCase,
		ListVideosUseCase:                 listVideosUseCase,
		UploadVideoUseCase:                uploadVideoUseCase,
		StreamVideoUseCase:                streamVideoUseCase,
		AcceleratorRegistry:               registry,
		AcceleratorControl:                controlServer,
		DeviceMonitor:                     deviceMonitor,
	}, nil
}

func (c *Container) Close(ctx context.Context) error {
	if c.AcceleratorControl != nil {
		stopCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		c.AcceleratorControl.Stop(stopCtx)
	}
	return nil
}
