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
	
	errChan := make(chan error, 2)
	
	go func() {
		log.Printf("Starting HTTP server on %s (hot_reload: %v, transport: %s)\n", 
			a.config.Server.HttpPort, a.config.Server.HotReloadEnabled, a.config.Stream.TransportFormat)
		if err := http.ListenAndServe(a.config.Server.HttpPort, mux); err != nil {
			errChan <- err
		}
	}()
	
	if a.config.Server.TLS.Enabled {
		go func() {
			log.Printf("Starting HTTPS server on %s (cert: %s)\n", 
				a.config.Server.HttpsPort, a.config.Server.TLS.CertFile)
			if err := http.ListenAndServeTLS(
				a.config.Server.HttpsPort, 
				a.config.Server.TLS.CertFile, 
				a.config.Server.TLS.KeyFile, 
				mux,
			); err != nil {
				errChan <- err
			}
		}()
	}
	
	return <-errChan
}

