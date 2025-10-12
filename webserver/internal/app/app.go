package app

import (
	"log"
	"net/http"

	"github.com/jrb/cuda-learning/webserver/internal/application"
	"github.com/jrb/cuda-learning/webserver/internal/config"
	"github.com/jrb/cuda-learning/webserver/internal/interfaces/connectrpc"
	"github.com/jrb/cuda-learning/webserver/internal/interfaces/static_http"
)

type App struct {
	config  *config.Config
	useCase *application.ProcessImageUseCase
}

func New(cfg *config.Config, useCase *application.ProcessImageUseCase) *App {
	return &App{
		config:  cfg,
		useCase: useCase,
	}
}

func (a *App) Run() error {
	mux := http.NewServeMux()
	
	rpcHandler := connectrpc.NewImageProcessorHandler(a.useCase)
	connectrpc.RegisterRoutesWithHandler(mux, rpcHandler)
	connectrpc.RegisterConfigService(mux, a.config)
	
	staticHandler := static_http.NewStaticHandler(a.config, rpcHandler)
	staticHandler.RegisterRoutes(mux)
	
	log.Printf("Starting server on %s (hot_reload: %v, transport: %s)\n", 
		a.config.Server.Port, a.config.Server.HotReloadEnabled, a.config.Stream.TransportFormat)
	return http.ListenAndServe(a.config.Server.Port, mux)
}

