package app

import (
	"context"
	"net/http"
	"sync"

	"connectrpc.com/connect"
	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"github.com/jrb/cuda-learning/webserver/pkg/config"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/logger"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/processor/loader"
	"github.com/jrb/cuda-learning/webserver/pkg/interfaces/connectrpc"
	httphandlers "github.com/jrb/cuda-learning/webserver/pkg/interfaces/http"
	"github.com/jrb/cuda-learning/webserver/pkg/interfaces/static_http"
	"github.com/jrb/cuda-learning/webserver/pkg/telemetry"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

type App struct {
	config                *config.Manager
	appContext            context.Context
	useCase               *application.ProcessImageUseCase
	getStreamConfigUC     *application.GetStreamConfigUseCase
	syncFlagsUC           *application.SyncFeatureFlagsUseCase
	listInputsUC          *application.ListInputsUseCase
	listAvailableImagesUC *application.ListAvailableImagesUseCase
	registry              *loader.Registry
	currentLoader         **loader.Loader
	loaderMutex           *sync.RWMutex
	interceptors          []connect.Interceptor
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

func WithListAvailableImagesUseCase(uc *application.ListAvailableImagesUseCase) AppOption {
	return func(a *App) {
		a.listAvailableImagesUC = uc
	}
}

func WithProcessorRegistry(registry *loader.Registry) AppOption {
	return func(a *App) {
		a.registry = registry
	}
}

func WithProcessorLoader(currentLoader **loader.Loader) AppOption {
	return func(a *App) {
		a.currentLoader = currentLoader
	}
}

func WithLoaderMutex(mu *sync.RWMutex) AppOption {
	return func(a *App) {
		a.loaderMutex = mu
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
		a.config.Observability.OtelCollectorEndpoint,
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
	if a.getStreamConfigUC != nil && a.syncFlagsUC != nil && a.listInputsUC != nil && a.listAvailableImagesUC != nil {
		connectrpc.RegisterConfigService(mux, a.getStreamConfigUC, a.syncFlagsUC, a.listInputsUC, a.listAvailableImagesUC,
			a.registry, a.currentLoader, a.loaderMutex, a.config, a.interceptors...)
	} else {
		logger.Global().Warn().Msg("Config service not registered (use cases unavailable)")
	}
}

func (a *App) setupHealthEndpoint(mux *http.ServeMux) {
	// Health endpoint uses plain HTTP instead of ConnectRPC because:
	// 1. Load balancers (k8s, Docker) require simple HTTP 200/503
	// 2. No protobuf complexity needed for basic health checks
	// 3. Industry standard for healthcheck endpoints
	healthHandler := httphandlers.NewHealthHandler()
	mux.Handle("/health", healthHandler)
	logger.Global().Info().Msg("Health endpoint registered at /health")
}

func (a *App) setupStaticHandler(mux *http.ServeMux) {
	staticHandler := static_http.NewStaticHandler(a.config.Server, a.config.Stream, a.useCase)
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
			Str("port", a.config.Server.HTTPPort).
			Bool("hot_reload", a.config.Server.HotReloadEnabled).
			Str("transport", a.config.Stream.TransportFormat).
			Msg("Starting HTTP server")
		if err := http.ListenAndServe(a.config.Server.HTTPPort, handler); err != nil {
			errChan <- err
		}
	}()

	if a.config.Server.TLS.Enabled {
		go func() {
			log.Info().
				Str("port", a.config.Server.HTTPSPort).
				Str("cert", a.config.Server.TLS.CertFile).
				Msg("Starting HTTPS server")
			if err := http.ListenAndServeTLS(
				a.config.Server.HTTPSPort,
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
