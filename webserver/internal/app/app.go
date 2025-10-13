package app

import (
	"log"
	"net/http"

	"github.com/jrb/cuda-learning/webserver/internal/application"
	"github.com/jrb/cuda-learning/webserver/internal/config"
	"github.com/jrb/cuda-learning/webserver/internal/interfaces/connectrpc"
	httphandlers "github.com/jrb/cuda-learning/webserver/internal/interfaces/http"
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
	
	if a.config.IsFeatureEnabled("enable_observability") {
		traceProxy := httphandlers.NewTraceProxyHandler(
			a.config.Observability.OtelCollectorEndpoint,
			true,
		)
		mux.Handle("/api/traces", traceProxy)
		log.Println("Trace proxy endpoint registered at /api/traces")
	}
	
	staticHandler := static_http.NewStaticHandler(a.config, a.useCase)
	staticHandler.RegisterRoutes(mux)
	
	var handler http.Handler = mux
	// Note: HTTP auto-instrumentation disabled - tracing works from handler level down
	if a.config.IsFeatureEnabled("enable_observability") {
		log.Println("OpenTelemetry enabled - tracing from handler level")
	}
	
	errChan := make(chan error, 2)
	
	go func() {
		log.Printf("Starting HTTP server on %s (hot_reload: %v, transport: %s)\n", 
			a.config.Server.HttpPort, a.config.Server.HotReloadEnabled, a.config.Stream.TransportFormat)
		if err := http.ListenAndServe(a.config.Server.HttpPort, handler); err != nil {
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
				handler,
			); err != nil {
				errChan <- err
			}
		}()
	}
	
	return <-errChan
}

