package app

import (
	"log"
	"net/http"

	"connectrpc.com/connect"
	"github.com/jrb/cuda-learning/webserver/internal/application"
	"github.com/jrb/cuda-learning/webserver/internal/config"
	"github.com/jrb/cuda-learning/webserver/internal/interfaces/connectrpc"
	httphandlers "github.com/jrb/cuda-learning/webserver/internal/interfaces/http"
	"github.com/jrb/cuda-learning/webserver/internal/interfaces/static_http"
	"github.com/jrb/cuda-learning/webserver/internal/telemetry"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
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

func (a *App) setupTelemetryMiddleware(handler http.Handler) http.Handler {
	if !a.config.IsFeatureEnabled("enable_observability") {
		return handler
	}

	instrumentedHandler := otelhttp.NewHandler(
		handler,
		"http-server",
		otelhttp.WithSpanNameFormatter(func(operation string, r *http.Request) string {
			return r.Method + " " + r.URL.Path
		}),
	)

	log.Println("OpenTelemetry HTTP instrumentation enabled")
	return instrumentedHandler
}

func (a *App) Run() error {
	mux := http.NewServeMux()
	
	var interceptors []connect.Interceptor
	if a.config.IsFeatureEnabled("enable_observability") {
		interceptors = append(interceptors, telemetry.TraceContextInterceptor())
		
		traceProxy := httphandlers.NewTraceProxyHandler(
			a.config.Observability.OtelCollectorEndpoint,
			true,
		)
		mux.Handle("/api/traces", traceProxy)
		log.Println("Trace proxy endpoint registered at /api/traces")
	}
	
	rpcHandler := connectrpc.NewImageProcessorHandler(a.useCase)
	connectrpc.RegisterRoutesWithHandler(mux, rpcHandler, interceptors...)
	connectrpc.RegisterConfigService(mux, a.config, interceptors...)
	
	staticHandler := static_http.NewStaticHandler(a.config, a.useCase)
	staticHandler.RegisterRoutes(mux)
	
	handler := a.setupTelemetryMiddleware(mux)
	
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

