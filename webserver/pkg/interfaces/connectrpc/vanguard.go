package connectrpc

import (
	"net/http"
	"sync"

	"connectrpc.com/connect"
	"connectrpc.com/vanguard"
	"github.com/jrb/cuda-learning/proto/gen/genconnect"
	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"github.com/jrb/cuda-learning/webserver/pkg/config"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/logger"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/processor/loader"
)

// VanguardConfig groups all dependencies needed to setup Vanguard transcoder
type VanguardConfig struct {
	ImageProcessorHandler *ImageProcessorHandler
	GetStreamConfigUC     *application.GetStreamConfigUseCase
	SyncFlagsUC           *application.SyncFeatureFlagsUseCase
	ListInputsUC          *application.ListInputsUseCase
	EvaluateFFUC          *application.EvaluateFeatureFlagUseCase
	GetSystemInfoUC       *application.GetSystemInfoUseCase
	Registry              *loader.Registry
	//TODO: loader needs a better abstraction
	CurrentLoader         **loader.Loader
	LoaderMutex           *sync.RWMutex
	ConfigManager         *config.Manager
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

	configHandler := NewConfigHandler(
		cfg.GetStreamConfigUC, cfg.SyncFlagsUC, cfg.ListInputsUC, cfg.EvaluateFFUC,
		cfg.GetSystemInfoUC, cfg.Registry, cfg.CurrentLoader, cfg.LoaderMutex, cfg.ConfigManager,
	)
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
