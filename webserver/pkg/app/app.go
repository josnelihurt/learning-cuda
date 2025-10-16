package app

import (
	"context"
	"net/http"

	"connectrpc.com/connect"
	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"github.com/jrb/cuda-learning/webserver/pkg/config"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/logger"
	"github.com/jrb/cuda-learning/webserver/pkg/interfaces/connectrpc"
	httphandlers "github.com/jrb/cuda-learning/webserver/pkg/interfaces/http"
	"github.com/jrb/cuda-learning/webserver/pkg/interfaces/static_http"
	"github.com/jrb/cuda-learning/webserver/pkg/telemetry"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

type App struct {
	config            *config.Manager
	appContext        context.Context
	useCase           *application.ProcessImageUseCase
	getStreamConfigUC *application.GetStreamConfigUseCase
	syncFlagsUC       *application.SyncFeatureFlagsUseCase
	listInputsUC      *application.ListInputsUseCase
	interceptors      []connect.Interceptor
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

func WithGetStreamConfigUseCase(uc *application.GetStreamConfigUseCase) AppOption {
	return func(a *App) {
		a.getStreamConfigUC = uc
	}
}

func WithSyncFlagsUseCase(uc *application.SyncFeatureFlagsUseCase) AppOption {
	return func(a *App) {
		a.syncFlagsUC = uc
	}
}

func WithListInputsUseCase(uc *application.ListInputsUseCase) AppOption {
	return func(a *App) {
		a.listInputsUC = uc
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

	logger.Global().Info().Msg("OpenTelemetry HTTP instrumentation enabled")
	return instrumentedHandler
}

func (a *App) setupObservability(mux *http.ServeMux) {
	log := logger.Global()
	if !a.config.IsObservabilityEnabled(a.appContext) {
		log.Info().Msg("OpenTelemetry HTTP instrumentation disabled")
		return
	}
	a.interceptors = append(a.interceptors, telemetry.TraceContextInterceptor())
	traceProxy := httphandlers.NewTraceProxyHandler(
		a.config.OtelCollectorEndpoint,
		true,
	)
	mux.Handle("/api/traces", traceProxy)
	log.Info().Msg("Trace proxy endpoint registered at /api/traces")
}

func (a *App) setupConnectRPCServices(mux *http.ServeMux) {
	{
		rpcHandler := connectrpc.NewImageProcessorHandler(a.useCase)
		connectrpc.RegisterRoutesWithHandler(mux, rpcHandler, a.interceptors...)
	}
	if a.getStreamConfigUC != nil && a.syncFlagsUC != nil && a.listInputsUC != nil {
		connectrpc.RegisterConfigService(mux, a.getStreamConfigUC, a.syncFlagsUC, a.listInputsUC, a.interceptors...)
	} else {
		logger.Global().Warn().Msg("Config service not registered (use cases unavailable)")
	}
}

func (a *App) setupHealthEndpoint(mux *http.ServeMux) {
	healthHandler := httphandlers.NewHealthHandler()
	mux.Handle("/health", healthHandler)
	logger.Global().Info().Msg("Health endpoint registered at /health")
}

func (a *App) setupStaticHandler(mux *http.ServeMux) {
	staticHandler := static_http.NewStaticHandler(a.config.ServerConfig, a.config.StreamConfig, a.useCase)
	staticHandler.RegisterRoutes(mux)
}

func (a *App) Run() error {
	log := logger.Global()
	mux := http.NewServeMux()
	a.setupObservability(mux)

	a.setupHealthEndpoint(mux)
	a.setupConnectRPCServices(mux)
	a.setupStaticHandler(mux)
	handler := a.makeTelemetryMiddleware(mux)

	errChan := make(chan error, 2)

	go func() {
		log.Info().
			Str("port", a.config.ServerConfig.HTTPPort).
			Bool("hot_reload", a.config.ServerConfig.HotReloadEnabled).
			Str("transport", a.config.StreamConfig.TransportFormat).
			Msg("Starting HTTP server")
		if err := http.ListenAndServe(a.config.ServerConfig.HTTPPort, handler); err != nil {
			errChan <- err
		}
	}()

	if a.config.ServerConfig.TLSConfig.Enabled {
		go func() {
			log.Info().
				Str("port", a.config.ServerConfig.HTTPSPort).
				Str("cert", a.config.ServerConfig.TLSConfig.CertFile).
				Msg("Starting HTTPS server")
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
