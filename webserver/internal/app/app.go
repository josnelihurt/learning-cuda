package app

import (
	"context"
	"log"
	"net/http"

	"connectrpc.com/connect"
	"github.com/jrb/cuda-learning/webserver/internal/application"
	"github.com/jrb/cuda-learning/webserver/internal/config"
	httpinfra "github.com/jrb/cuda-learning/webserver/internal/infrastructure/http"
	"github.com/jrb/cuda-learning/webserver/internal/interfaces/connectrpc"
	httphandlers "github.com/jrb/cuda-learning/webserver/internal/interfaces/http"
	"github.com/jrb/cuda-learning/webserver/internal/interfaces/static_http"
	"github.com/jrb/cuda-learning/webserver/internal/telemetry"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

type App struct {
	config       *config.Manager
	appContext   context.Context
	useCase      *application.ProcessImageUseCase
	interceptors []connect.Interceptor
}

type AppOption func(*App)

func New(appContext context.Context, opts ...AppOption) *App {
	app := &App{
		appContext: appContext,
	}

	for _, opt := range opts {
		opt(app)
	}

	return app
}

func WithConfig(cfg *config.Manager) AppOption {
	return func(a *App) {
		a.config = cfg
	}
}

func WithUseCase(useCase *application.ProcessImageUseCase) AppOption {
	return func(a *App) {
		a.useCase = useCase
	}
}

func (a *App) makeTelemetryMiddleware(handler http.Handler) http.Handler {
	if !a.config.IsObservabilityEnabled(a.appContext) {
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

func (a *App) setupObservability(mux *http.ServeMux) {
	if !a.config.IsObservabilityEnabled(a.appContext) {
		log.Println("OpenTelemetry HTTP instrumentation disabled")
		return
	}
	a.interceptors = append(a.interceptors, telemetry.TraceContextInterceptor())
	traceProxy := httphandlers.NewTraceProxyHandler(
		a.config.OtelCollectorEndpoint,
		true,
	)
	mux.Handle("/api/traces", traceProxy)
	log.Println("Trace proxy endpoint registered at /api/traces")
}

func (a *App) setupConnectRPCServices(mux *http.ServeMux) {
	{
		rpcHandler := connectrpc.NewImageProcessorHandler(a.useCase)
		connectrpc.RegisterRoutesWithHandler(mux, rpcHandler, a.interceptors...)
	}
	{
		httpClient := httpinfra.New(&http.Client{
			Timeout: a.config.HttpClientTimeout,
		})
		connectrpc.RegisterConfigService(mux, a.config.StreamConfig, a.config, httpClient, a.interceptors...)
	}
}

func (a *App) setupStaticHandler(mux *http.ServeMux) {
	staticHandler := static_http.NewStaticHandler(a.config.ServerConfig, a.config.StreamConfig, a.useCase)
	staticHandler.RegisterRoutes(mux)
}

func (a *App) Run() error {
	mux := http.NewServeMux()
	a.setupObservability(mux)

	a.setupConnectRPCServices(mux)
	a.setupStaticHandler(mux)
	handler := a.makeTelemetryMiddleware(mux)

	errChan := make(chan error, 2)

	go func() {
		log.Printf("Starting HTTP server on %s (hot_reload: %v, transport: %s)\n",
			a.config.ServerConfig.HTTPPort, a.config.ServerConfig.HotReloadEnabled, a.config.StreamConfig.TransportFormat)
		if err := http.ListenAndServe(a.config.ServerConfig.HTTPPort, handler); err != nil {
			errChan <- err
		}
	}()

	if a.config.ServerConfig.TLSConfig.Enabled {
		go func() {
			log.Printf("Starting HTTPS server on %s (cert: %s)\n",
				a.config.ServerConfig.HTTPSPort, a.config.ServerConfig.TLSConfig.CertFile)
			if err := http.ListenAndServeTLS(
				a.config.ServerConfig.HTTPSPort,
				a.config.ServerConfig.TLSConfig.CertFile,
				a.config.ServerConfig.TLSConfig.KeyFile,
				handler,
			); err != nil {
				errChan <- err
			}
		}()
	}

	return <-errChan
}
