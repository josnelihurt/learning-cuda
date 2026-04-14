package app

import (
	"context"
	"errors"
	"net/http"

	"connectrpc.com/connect"
	"github.com/jrb/cuda-learning/src/go_api/pkg/application"
	"github.com/jrb/cuda-learning/src/go_api/pkg/config"
	"github.com/jrb/cuda-learning/src/go_api/pkg/domain"
	domainInterfaces "github.com/jrb/cuda-learning/src/go_api/pkg/domain/interfaces"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/logger"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/processor"
	"github.com/jrb/cuda-learning/src/go_api/pkg/interfaces/connectrpc"
	httphandlers "github.com/jrb/cuda-learning/src/go_api/pkg/interfaces/http"
	"github.com/jrb/cuda-learning/src/go_api/pkg/interfaces/websocket"
	"github.com/jrb/cuda-learning/src/go_api/pkg/telemetry"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

type App struct {
	config                *config.Manager
	appContext            context.Context
	useCase               *application.ProcessImageUseCase
	grpcProcessor         domain.ImageProcessor
	grpcProcessorClient   *processor.GRPCClient
	processorCapsUC       application.ProcessorCapabilitiesUseCase
	getStreamConfigUC     *application.GetStreamConfigUseCase
	getSystemInfoUC       *application.GetSystemInfoUseCase
	featureFlagRepo       domain.FeatureFlagRepository
	listInputsUC          *application.ListInputsUseCase
	evaluateFFUC          *application.EvaluateFeatureFlagUseCase
	listAvailableImagesUC *application.ListAvailableImagesUseCase
	uploadImageUC         *application.UploadImageUseCase
	listVideosUC          *application.ListVideosUseCase
	uploadVideoUC         *application.UploadVideoUseCase
	videoRepository       domain.VideoRepository
	deviceMonitor         domainInterfaces.MQTTDeviceMonitor
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

func WithGRPCProcessor(proc domain.ImageProcessor) Option {
	return func(a *App) {
		a.grpcProcessor = proc
	}
}

func WithGRPCProcessorClient(client *processor.GRPCClient) Option {
	return func(a *App) {
		a.grpcProcessorClient = client
	}
}

func WithProcessorCapabilitiesUseCase(uc application.ProcessorCapabilitiesUseCase) Option {
	return func(a *App) {
		a.processorCapsUC = uc
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

func WithFeatureFlagRepository(repo domain.FeatureFlagRepository) Option {
	return func(a *App) {
		a.featureFlagRepo = repo
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

func WithDeviceMonitor(monitor domainInterfaces.MQTTDeviceMonitor) Option {
	return func(a *App) {
		a.deviceMonitor = monitor
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
		a.config.Observability.OtelCollectorHTTPEndpoint,
		true,
	)
	mux.Handle("/api/traces", traceProxy)
	log.Info().Msg("Trace proxy endpoint registered at /api/traces")

	logsProxy := httphandlers.NewLogsProxyHandler(
		a.config.Observability.OtelCollectorHTTPEndpoint,
		true,
	)
	mux.Handle("/api/logs", logsProxy)
	log.Info().Msg("Logs proxy endpoint registered at /api/logs")
}

func (a *App) setupConnectRPCServices(mux *http.ServeMux) {
	rpcHandler := connectrpc.NewImageProcessorHandlerWithGRPC(
		a.useCase,
		a.processorCapsUC,
		a.evaluateFFUC,
		a.grpcProcessorClient,
	)

	connectrpc.RegisterConfigService(
		mux,
		connectrpc.ConfigHandlerDeps{
			GetStreamConfigUC: a.getStreamConfigUC,
			FeatureFlagRepo:   a.featureFlagRepo,
			ListInputsUC:      a.listInputsUC,
			EvaluateFFUC:      a.evaluateFFUC,
			GetSystemInfoUC:   a.getSystemInfoUC,
			ConfigManager:     a.config,
			ProcessorCapsUC:   a.processorCapsUC,
		},
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

	connectrpc.RegisterRemoteManagementService(
		mux,
		a.grpcProcessorClient,
		a.config,
		a.deviceMonitor,
		a.interceptors...,
	)

	transcoder := connectrpc.SetupVanguardTranscoder(&connectrpc.VanguardConfig{
		ImageProcessorHandler: rpcHandler,
		GetStreamConfigUC:     a.getStreamConfigUC,
		FeatureFlagRepo:       a.featureFlagRepo,
		ListInputsUC:          a.listInputsUC,
		EvaluateFFUC:          a.evaluateFFUC,
		GetSystemInfoUC:       a.getSystemInfoUC,
		ConfigManager:         a.config,
		ProcessorCapsUC:       a.processorCapsUC,
		ListAvailableImagesUC: a.listAvailableImagesUC,
		UploadImageUC:         a.uploadImageUC,
		ListVideosUC:          a.listVideosUC,
		UploadVideoUC:         a.uploadVideoUC,
		Interceptors:          a.interceptors,
	})

	mux.Handle("/api/", transcoder)

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

func (a *App) setupWebSocketHandler(mux *http.ServeMux) {
	wsHandler := websocket.NewHandler(a.useCase, a.config.Stream, a.videoRepository, a.evaluateFFUC, a.grpcProcessor)
	mux.HandleFunc("/ws", wsHandler.HandleWebSocket)
	logger.Global().Info().Msg("WebSocket endpoint registered at /ws")
}

func (a *App) setupWebRTCSignalingWebSocket(mux *http.ServeMux) {
	webrtcHandler := websocket.NewWebRTCSignalingHandler(a.grpcProcessorClient)
	mux.HandleFunc("/ws/webrtc-signaling", webrtcHandler.HandleWebRTCSignaling)
	logger.Global().Info().Msg("WebRTC signaling WebSocket endpoint registered at /ws/webrtc-signaling")
}

func (a *App) Run() error {
	log := logger.Global()
	defer func() {
		if err := a.deviceMonitor.Stop(); err != nil {
			log.Warn().Err(err).Msg("Failed to stop MQTT device monitor")
		}
	}()

	if a.deviceMonitor == nil {
		return errors.New("MQTT device monitor not initialized")
	}

	if err := a.deviceMonitor.Start(a.appContext); err != nil {
		log.Err(err).Msg("Failed to start MQTT device monitor")
		return err
	}
	log.Info().Msg("MQTT device monitor started")

	mux := http.NewServeMux()
	a.setupObservability(mux)

	a.setupHealthEndpoint(mux)
	a.setupWebSocketHandler(mux)
	a.setupWebRTCSignalingWebSocket(mux)
	a.setupConnectRPCServices(mux)
	handler := a.makeTelemetryMiddleware(mux)

	errChan := make(chan error, 2)

	go func() {
		log.Info().
			Str("port", a.config.Server.HTTPPort).
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
