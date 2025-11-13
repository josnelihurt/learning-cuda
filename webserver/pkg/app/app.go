package app

import (
	"context"
	"net/http"
	"strings"
	"sync"

	"connectrpc.com/connect"
	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"github.com/jrb/cuda-learning/webserver/pkg/config"
	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/logger"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/processor"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/processor/loader"
	"github.com/jrb/cuda-learning/webserver/pkg/interfaces/connectrpc"
	httphandlers "github.com/jrb/cuda-learning/webserver/pkg/interfaces/http"
	"github.com/jrb/cuda-learning/webserver/pkg/interfaces/statichttp"
	"github.com/jrb/cuda-learning/webserver/pkg/telemetry"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

type App struct {
	config                *config.Manager
	appContext            context.Context
	useCase               *application.ProcessImageUseCase
	getStreamConfigUC     *application.GetStreamConfigUseCase
	getSystemInfoUC       *application.GetSystemInfoUseCase
	syncFlagsUC           *application.SyncFeatureFlagsUseCase
	listInputsUC          *application.ListInputsUseCase
	evaluateFFUC          *application.EvaluateFeatureFlagUseCase
	listAvailableImagesUC *application.ListAvailableImagesUseCase
	uploadImageUC         *application.UploadImageUseCase
	listVideosUC          *application.ListVideosUseCase
	uploadVideoUC         *application.UploadVideoUseCase
	videoRepository       domain.VideoRepository
	registry              *loader.Registry
	currentLoader         **loader.Loader
	loaderMutex           *sync.RWMutex
	cppConnector          *processor.CppConnector
	interceptors          []connect.Interceptor
}

type Option func(*App)

func New(appContext context.Context, opts ...Option) *App {
	app := &App{
		appContext: appContext,
	}

	for _, opt := range opts {
		opt(app)
	}

	return app
}

func WithConfig(cfg *config.Manager) Option {
	return func(a *App) {
		a.config = cfg
	}
}

func WithUseCase(useCase *application.ProcessImageUseCase) Option {
	return func(a *App) {
		a.useCase = useCase
	}
}

func WithGetStreamConfigUseCase(uc *application.GetStreamConfigUseCase) Option {
	return func(a *App) {
		a.getStreamConfigUC = uc
	}
}

func WithGetSystemInfoUseCase(uc *application.GetSystemInfoUseCase) Option {
	return func(a *App) {
		a.getSystemInfoUC = uc
	}
}

func WithSyncFlagsUseCase(uc *application.SyncFeatureFlagsUseCase) Option {
	return func(a *App) {
		a.syncFlagsUC = uc
	}
}

func WithListInputsUseCase(uc *application.ListInputsUseCase) Option {
	return func(a *App) {
		a.listInputsUC = uc
	}
}

func WithEvaluateFFUseCase(uc *application.EvaluateFeatureFlagUseCase) Option {
	return func(a *App) {
		a.evaluateFFUC = uc
	}
}

func WithListAvailableImagesUseCase(uc *application.ListAvailableImagesUseCase) Option {
	return func(a *App) {
		a.listAvailableImagesUC = uc
	}
}

func WithUploadImageUseCase(uc *application.UploadImageUseCase) Option {
	return func(a *App) {
		a.uploadImageUC = uc
	}
}

func WithListVideosUseCase(uc *application.ListVideosUseCase) Option {
	return func(a *App) {
		a.listVideosUC = uc
	}
}

func WithUploadVideoUseCase(uc *application.UploadVideoUseCase) Option {
	return func(a *App) {
		a.uploadVideoUC = uc
	}
}

func WithVideoRepository(repo domain.VideoRepository) Option {
	return func(a *App) {
		a.videoRepository = repo
	}
}

func WithProcessorRegistry(registry *loader.Registry) Option {
	return func(a *App) {
		a.registry = registry
	}
}

func WithProcessorLoader(currentLoader **loader.Loader) Option {
	return func(a *App) {
		a.currentLoader = currentLoader
	}
}

func WithLoaderMutex(mu *sync.RWMutex) Option {
	return func(a *App) {
		a.loaderMutex = mu
	}
}

func WithCppConnector(connector *processor.CppConnector) Option {
	return func(a *App) {
		a.cppConnector = connector
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

	logsProxy := httphandlers.NewLogsProxyHandler(
		a.config.Observability.OtelCollectorEndpoint,
		true,
	)
	mux.Handle("/api/logs", logsProxy)
	log.Info().Msg("Logs proxy endpoint registered at /api/logs")
}

func (a *App) setupConnectRPCServices(mux *http.ServeMux) {
	rpcHandler := connectrpc.NewImageProcessorHandler(a.useCase)

	connectrpc.RegisterConfigService(
		mux,
		a.getStreamConfigUC,
		a.syncFlagsUC,
		a.listInputsUC,
		a.evaluateFFUC,
		a.getSystemInfoUC,
		a.config,
		a.cppConnector,
		a.interceptors...,
	)

	connectrpc.RegisterFileService(
		mux,
		a.listAvailableImagesUC,
		a.uploadImageUC,
		a.listVideosUC,
		a.uploadVideoUC,
		a.interceptors...,
	)

	connectrpc.RegisterRoutesWithHandler(mux, rpcHandler, a.interceptors...)

	transcoder := connectrpc.SetupVanguardTranscoder(&connectrpc.VanguardConfig{
		ImageProcessorHandler: rpcHandler,
		GetStreamConfigUC:     a.getStreamConfigUC,
		SyncFlagsUC:           a.syncFlagsUC,
		ListInputsUC:          a.listInputsUC,
		EvaluateFFUC:          a.evaluateFFUC,
		GetSystemInfoUC:       a.getSystemInfoUC,
		Registry:              a.registry,
		CurrentLoader:         a.currentLoader,
		LoaderMutex:           a.loaderMutex,
		ConfigManager:         a.config,
		CppConnector:          a.cppConnector,
		ListAvailableImagesUC: a.listAvailableImagesUC,
		UploadImageUC:         a.uploadImageUC,
		ListVideosUC:          a.listVideosUC,
		UploadVideoUC:         a.uploadVideoUC,
		Interceptors:          a.interceptors,
	})

	staticHandler := statichttp.NewStaticHandler(&a.config.Server, a.config.Stream, a.useCase, a.videoRepository, a.config.Flipt.URL)
	serveIndex := staticHandler.GetServeIndex()

	// Register catch-all handler AFTER Connect-RPC handlers to ensure specific routes are matched first
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// REST API routes handled by transcoder
		if strings.HasPrefix(r.URL.Path, "/api/") {
			transcoder.ServeHTTP(w, r)
			return
		}

		// Connect-RPC routes should already be handled by registered handlers above
		// Vanguard transcoder doesn't support direct Connect-RPC GET requests,
		// so we should NOT fallback to transcoder for Connect-RPC paths.
		// The registered Connect-RPC handlers handle both GET and POST.
		if strings.HasPrefix(r.URL.Path, "/cuda_learning.") ||
			strings.HasPrefix(r.URL.Path, "/com.jrb.") {
			// These should be handled by Connect-RPC handlers, not transcoder
			// If we get here, it means no handler matched - return 404
			http.NotFound(w, r)
			return
		}

		// For all other routes, serve the SPA index
		serveIndex(w, r)
	})

	logger.Global().Info().Msg("Connect-RPC handlers and Vanguard transcoder registered (REST + Connect + gRPC)")
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
	staticHandler := statichttp.NewStaticHandler(&a.config.Server, a.config.Stream, a.useCase, a.videoRepository, a.config.Flipt.URL)
	staticHandler.RegisterRoutes(mux)
}

func (a *App) Run() error {
	log := logger.Global()
	mux := http.NewServeMux()
	a.setupObservability(mux)

	a.setupHealthEndpoint(mux)
	a.setupStaticHandler(mux)
	a.setupConnectRPCServices(mux)
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
