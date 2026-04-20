package connectrpc

import (
	"net/http"

	"connectrpc.com/connect"
	"connectrpc.com/vanguard"
	"github.com/jrb/cuda-learning/proto/gen/genconnect"
	imageapp "github.com/jrb/cuda-learning/src/go_api/pkg/application/media/image"
	videoapp "github.com/jrb/cuda-learning/src/go_api/pkg/application/media/video"
	systemapp "github.com/jrb/cuda-learning/src/go_api/pkg/application/platform/system"
	"github.com/jrb/cuda-learning/src/go_api/pkg/config"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/logger"
)

// VanguardConfig groups all dependencies needed to setup Vanguard transcoder
type VanguardConfig struct {
	ImageProcessorHandler *ImageProcessorHandler
	FeatureFlagRepo       featureFlagRepository
	ListInputsUC          useCase[videoapp.ListInputsUseCaseInput, videoapp.ListInputsUseCaseOutput]
	GetSystemInfoUC       useCase[systemapp.GetSystemInfoUseCaseInput, systemapp.GetSystemInfoUseCaseOutput]
	ConfigManager         *config.Manager
	ProcessorCapsUC       processorCapabilitiesUseCase
	ListAvailableImagesUC useCase[imageapp.ListAvailableImagesUseCaseInput, imageapp.ListAvailableImagesUseCaseOutput]
	UploadImageUC         useCase[imageapp.UploadImageUseCaseInput, imageapp.UploadImageUseCaseOutput]
	ListVideosUC          useCase[videoapp.ListVideosUseCaseInput, videoapp.ListVideosUseCaseOutput]
	UploadVideoUC         useCase[videoapp.UploadVideoUseCaseInput, videoapp.UploadVideoUseCaseOutput]
	Interceptors          []connect.Interceptor
}

// SetupVanguardTranscoder configures the Vanguard transcoder with all the services
func SetupVanguardTranscoder(cfg *VanguardConfig) http.Handler {
	log := logger.Global()

	var opts []connect.HandlerOption
	if len(cfg.Interceptors) > 0 {
		opts = append(opts, connect.WithInterceptors(cfg.Interceptors...))
	}

	_, imageProcessorConnectHandler := genconnect.NewImageProcessorServiceHandler(
		cfg.ImageProcessorHandler, opts...,
	)

	configHandler := NewConfigHandler(ConfigHandlerDeps{
		FeatureFlagRepo: cfg.FeatureFlagRepo,
		ListInputsUC:    cfg.ListInputsUC,
		GetSystemInfoUC: cfg.GetSystemInfoUC,
		ConfigManager:   cfg.ConfigManager,
		ProcessorCapsUC: cfg.ProcessorCapsUC,
	})
	_, configConnectHandler := genconnect.NewConfigServiceHandler(configHandler, opts...)

	fileHandler := NewFileHandler(
		cfg.ListAvailableImagesUC, cfg.UploadImageUC, cfg.ListVideosUC, cfg.UploadVideoUC,
	)
	_, fileConnectHandler := genconnect.NewFileServiceHandler(fileHandler, opts...)

	services := []*vanguard.Service{
		vanguard.NewService(genconnect.ImageProcessorServiceName, imageProcessorConnectHandler),
		vanguard.NewService(genconnect.ConfigServiceName, configConnectHandler),
		vanguard.NewService(genconnect.FileServiceName, fileConnectHandler),
	}

	transcoder, err := vanguard.NewTranscoder(services)
	if err != nil {
		log.Error().
			Err(err).
			Msg("Failed to create Vanguard transcoder")
		panic("failed to create vanguard transcoder: " + err.Error())
	}

	return transcoder
}
