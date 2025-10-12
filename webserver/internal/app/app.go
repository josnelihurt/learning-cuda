package app

import (
	"log"
	"net/http"

	"github.com/jrb/cuda-learning/webserver/internal/application"
	"github.com/jrb/cuda-learning/webserver/internal/interfaces/connectrpc"
	"github.com/jrb/cuda-learning/webserver/internal/interfaces/static_http"
)

type App struct {
	config  *Config
	useCase *application.ProcessImageUseCase
}

func New(config *Config, useCase *application.ProcessImageUseCase) *App {
	return &App{
		config:  config,
		useCase: useCase,
	}
}

func (a *App) Run() error {
	mux := http.NewServeMux()
	
	rpcHandler := connectrpc.NewImageProcessorHandler(a.useCase)
	connectrpc.RegisterRoutesWithHandler(mux, rpcHandler)
	
	staticHandler := static_http.NewStaticHandler(a.config.WebRootPath, a.config.DevMode, rpcHandler)
	staticHandler.RegisterRoutes(mux)
	
	log.Printf("Starting server on %s\n", a.config.Port)
	return http.ListenAndServe(a.config.Port, mux)
}

