package connectrpc

import (
	"net/http"
	"sync"

	"connectrpc.com/connect"
	"github.com/jrb/cuda-learning/proto/gen/genconnect"
	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/processor/loader"
)

func RegisterRoutes(mux *http.ServeMux, useCase *application.ProcessImageUseCase) {
	handler := NewImageProcessorHandler(useCase)
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
	registry *loader.Registry,
	currentLoader **loader.Loader,
	loaderMutex *sync.RWMutex,
	interceptors ...connect.Interceptor,
) {
	configHandler := NewConfigHandler(getStreamConfigUC, syncFlagsUC, listInputsUC, registry, currentLoader, loaderMutex)

	var opts []connect.HandlerOption
	if len(interceptors) > 0 {
		opts = append(opts, connect.WithInterceptors(interceptors...))
	}

	path, rpcHandler := genconnect.NewConfigServiceHandler(configHandler, opts...)
	mux.Handle(path, rpcHandler)
}
