package connectrpc

import (
	"net/http"

	"connectrpc.com/connect"
	"github.com/jrb/cuda-learning/proto/gen/genconnect"
	imageapp "github.com/jrb/cuda-learning/src/go_api/pkg/application/media/image"
	videoapp "github.com/jrb/cuda-learning/src/go_api/pkg/application/media/video"
	"github.com/jrb/cuda-learning/src/go_api/pkg/config"
	domainInterfaces "github.com/jrb/cuda-learning/src/go_api/pkg/domain/interfaces"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/processor"
)

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
	deps ConfigHandlerDeps,
	interceptors ...connect.Interceptor,
) {
	configHandler := NewConfigHandler(deps)

	var opts []connect.HandlerOption
	if len(interceptors) > 0 {
		opts = append(opts, connect.WithInterceptors(interceptors...))
	}

	path, rpcHandler := genconnect.NewConfigServiceHandler(configHandler, opts...)
	mux.Handle(path, rpcHandler)
}

func RegisterFileService(
	mux *http.ServeMux,
	listAvailableImagesUC *imageapp.ListAvailableImagesUseCase,
	uploadImageUC *imageapp.UploadImageUseCase,
	listVideosUC *videoapp.ListVideosUseCase,
	uploadVideoUC *videoapp.UploadVideoUseCase,
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

func RegisterWebRTCSignalingService(
	mux *http.ServeMux,
	client WebRTCSignalingClient,
	interceptors ...connect.Interceptor,
) {
	handler := NewWebRTCSignalingHandler(client)

	var opts []connect.HandlerOption
	if len(interceptors) > 0 {
		opts = append(opts, connect.WithInterceptors(interceptors...))
	}

	path, rpcHandler := genconnect.NewWebRTCSignalingServiceHandler(handler, opts...)
	mux.Handle(path, rpcHandler)
}

func RegisterRemoteManagementService(
	mux *http.ServeMux,
	gateway *processor.AcceleratorGateway,
	configManager *config.Manager,
	deviceMonitor domainInterfaces.MQTTDeviceMonitor,
	interceptors ...connect.Interceptor,
) {
	handler := NewRemoteManagementHandler(gateway, configManager, deviceMonitor)

	var opts []connect.HandlerOption
	if len(interceptors) > 0 {
		opts = append(opts, connect.WithInterceptors(interceptors...))
	}

	path, rpcHandler := genconnect.NewRemoteManagementServiceHandler(handler, opts...)
	mux.Handle(path, rpcHandler)
}
