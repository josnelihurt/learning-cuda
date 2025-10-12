package connectrpc

import (
	"net/http"

	"github.com/jrb/cuda-learning/proto/gen/genconnect"
	"github.com/jrb/cuda-learning/webserver/internal/application"
	"github.com/jrb/cuda-learning/webserver/internal/config"
)

func RegisterRoutes(mux *http.ServeMux, useCase *application.ProcessImageUseCase) {
	handler := NewImageProcessorHandler(useCase)
	path, rpcHandler := genconnect.NewImageProcessorServiceHandler(handler)
	mux.Handle(path, rpcHandler)
}

func RegisterRoutesWithHandler(mux *http.ServeMux, handler *ImageProcessorHandler) {
	path, rpcHandler := genconnect.NewImageProcessorServiceHandler(handler)
	mux.Handle(path, rpcHandler)
}

func RegisterConfigService(mux *http.ServeMux, cfg *config.Config) {
	configHandler := NewConfigHandler(cfg)
	path, rpcHandler := genconnect.NewConfigServiceHandler(configHandler)
	mux.Handle(path, rpcHandler)
}

