package connectrpc

import (
	"net/http"

	"connectrpc.com/connect"
	"github.com/jrb/cuda-learning/proto/gen/genconnect"
	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"github.com/jrb/cuda-learning/webserver/pkg/config"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/processor"
)

func RegisterRoutes(
	mux *http.ServeMux,
	useCase *application.ProcessImageUseCase,
	capabilities filterCapabilitiesProvider,
) {
	handler := NewImageProcessorHandler(useCase, capabilities)
	path, rpcHandler := genconnect.NewImageProcessorServiceHandler(handler)
	mux.Handle(path, rpcHandler)
}

func RegisterRoutesWithHandler(mux *http.ServeMux, handler *ImageProcessorHandler, interceptors ...connect.Interceptor) {
	var opts []connect.HandlerOption
	if len(interceptors) > 0 {
		opts = append(opts, connect.WithInterceptors(interceptors...))
	}

	path, rpcHandler := genconnect.NewImageProcessorServiceHandler(handler, opts...)
	mux.Handle(path, rpcHandler)
}

func RegisterConfigService(
	mux *http.ServeMux,
	getStreamConfigUC *application.GetStreamConfigUseCase,
	syncFlagsUC *application.SyncFeatureFlagsUseCase,
	listInputsUC *application.ListInputsUseCase,
	evaluateFFUC *application.EvaluateFeatureFlagUseCase,
	getSystemInfoUC *application.GetSystemInfoUseCase,
	configManager *config.Manager,
	cppConnector interface{},
	interceptors ...connect.Interceptor,
) {
	var connector *processor.CppConnector
	if cppConnector != nil {
		if c, ok := cppConnector.(*processor.CppConnector); ok {
			connector = c
		}
	}
	configHandler := NewConfigHandler(getStreamConfigUC, syncFlagsUC, listInputsUC, evaluateFFUC, getSystemInfoUC, configManager, connector)

	var opts []connect.HandlerOption
	if len(interceptors) > 0 {
		opts = append(opts, connect.WithInterceptors(interceptors...))
	}

	path, rpcHandler := genconnect.NewConfigServiceHandler(configHandler, opts...)
	mux.Handle(path, rpcHandler)
}

func RegisterFileService(
	mux *http.ServeMux,
	listAvailableImagesUC *application.ListAvailableImagesUseCase,
	uploadImageUC *application.UploadImageUseCase,
	listVideosUC *application.ListVideosUseCase,
	uploadVideoUC *application.UploadVideoUseCase,
	interceptors ...connect.Interceptor,
) {
	fileHandler := NewFileHandler(listAvailableImagesUC, uploadImageUC, listVideosUC, uploadVideoUC)

	var opts []connect.HandlerOption
	if len(interceptors) > 0 {
		opts = append(opts, connect.WithInterceptors(interceptors...))
	}

	path, rpcHandler := genconnect.NewFileServiceHandler(fileHandler, opts...)
	mux.Handle(path, rpcHandler)
}
