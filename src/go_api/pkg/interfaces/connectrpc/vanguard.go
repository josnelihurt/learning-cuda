package connectrpc

import (
	"net/http"

	"connectrpc.com/connect"
	"connectrpc.com/vanguard"
	"github.com/jrb/cuda-learning/proto/gen/genconnect"
	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"github.com/jrb/cuda-learning/webserver/pkg/config"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/logger"
)

// VanguardConfig groups all dependencies needed to setup Vanguard transcoder
type VanguardConfig struct {
	ImageProcessorHandler *ImageProcessorHandler
	GetStreamConfigUC     *application.GetStreamConfigUseCase
	SyncFlagsUC           *application.SyncFeatureFlagsUseCase
	ListInputsUC          *application.ListInputsUseCase
	EvaluateFFUC          *application.EvaluateFeatureFlagUseCase
	GetSystemInfoUC       *application.GetSystemInfoUseCase
	ConfigManager         *config.Manager
	ProcessorCapsUC       application.ProcessorCapabilitiesUseCase
	ListAvailableImagesUC *application.ListAvailableImagesUseCase
	UploadImageUC         *application.UploadImageUseCase
	ListVideosUC          *application.ListVideosUseCase
	UploadVideoUC         *application.UploadVideoUseCase
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
		GetStreamConfigUC: cfg.GetStreamConfigUC,
		SyncFlagsUC:       cfg.SyncFlagsUC,
		ListInputsUC:      cfg.ListInputsUC,
		EvaluateFFUC:      cfg.EvaluateFFUC,
		GetSystemInfoUC:   cfg.GetSystemInfoUC,
		ConfigManager:     cfg.ConfigManager,
		ProcessorCapsUC:   cfg.ProcessorCapsUC,
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
