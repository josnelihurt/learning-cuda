package app

import (
	"context"
	"errors"
	"net/http"

	"connectrpc.com/connect"
	ffapp "github.com/jrb/cuda-learning/src/go_api/pkg/application/flags"
	imageapp "github.com/jrb/cuda-learning/src/go_api/pkg/application/media/image"
	videoapp "github.com/jrb/cuda-learning/src/go_api/pkg/application/media/video"
	systemapp "github.com/jrb/cuda-learning/src/go_api/pkg/application/platform/system"
	"github.com/jrb/cuda-learning/src/go_api/pkg/config"
	"github.com/jrb/cuda-learning/src/go_api/pkg/domain"
	domainInterfaces "github.com/jrb/cuda-learning/src/go_api/pkg/domain/interfaces"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/logger"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/processor"
	"github.com/jrb/cuda-learning/src/go_api/pkg/interfaces/connectrpc"
	httphandlers "github.com/jrb/cuda-learning/src/go_api/pkg/interfaces/http"
	"github.com/jrb/cuda-learning/src/go_api/pkg/telemetry"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

type App struct {
	config                *config.Manager
	appContext            context.Context
	useCase               *imageapp.ProcessImageUseCase
	acceleratorClient     *processor.GRPCClient
	processorCapsUC       *systemapp.ProcessorCapabilitiesUseCase
	getSystemInfoUC       *systemapp.GetSystemInfoUseCase
	featureFlagRepo       domain.FeatureFlagRepository
	listInputsUC          *videoapp.ListInputsUseCase
	evaluateFFUC          *ffapp.EvaluateFeatureFlagUseCase
	streamVideoUC         *videoapp.StreamVideoUseCase
	listAvailableImagesUC *imageapp.ListAvailableImagesUseCase
	uploadImageUC         *imageapp.UploadImageUseCase
	listVideosUC          *videoapp.ListVideosUseCase
	uploadVideoUC         *videoapp.UploadVideoUseCase
	videoRepository       domain.VideoRepository
	deviceMonitor         domainInterfaces.MQTTDeviceMonitor
	interceptors          []connect.Interceptor
}

type Deps struct {
	Config                *config.Manager
	UseCase               *imageapp.ProcessImageUseCase
	AcceleratorClient     *processor.GRPCClient
	ProcessorCapsUC       *systemapp.ProcessorCapabilitiesUseCase
	GetSystemInfoUC       *systemapp.GetSystemInfoUseCase
	FeatureFlagRepo       domain.FeatureFlagRepository
	ListInputsUC          *videoapp.ListInputsUseCase
	EvaluateFFUC          *ffapp.EvaluateFeatureFlagUseCase
	StreamVideoUC         *videoapp.StreamVideoUseCase
	ListAvailableImagesUC *imageapp.ListAvailableImagesUseCase
	UploadImageUC         *imageapp.UploadImageUseCase
	ListVideosUC          *videoapp.ListVideosUseCase
	UploadVideoUC         *videoapp.UploadVideoUseCase
	VideoRepository       domain.VideoRepository
	DeviceMonitor         domainInterfaces.MQTTDeviceMonitor
}

func New(appContext context.Context, deps Deps, opts ...Option) (*App, error) {
	if deps.Config == nil {
		return nil, errors.New("config is required")
	}
	if deps.UseCase == nil {
		return nil, errors.New("process image use case is required")
	}
	if deps.AcceleratorClient == nil {
		return nil, errors.New("accelerator client is required")
	}
	if deps.ProcessorCapsUC == nil {
		return nil, errors.New("processor capabilities use case is required")
	}
	if deps.GetSystemInfoUC == nil {
		return nil, errors.New("get system info use case is required")
	}
	if deps.FeatureFlagRepo == nil {
		return nil, errors.New("feature flag repository is required")
	}
	if deps.ListInputsUC == nil {
		return nil, errors.New("list inputs use case is required")
	}
	if deps.EvaluateFFUC == nil {
		return nil, errors.New("evaluate feature flag use case is required")
	}
	if deps.StreamVideoUC == nil {
		return nil, errors.New("stream video use case is required")
	}
	if deps.ListAvailableImagesUC == nil {
		return nil, errors.New("list available images use case is required")
	}
	if deps.UploadImageUC == nil {
		return nil, errors.New("upload image use case is required")
	}
	if deps.ListVideosUC == nil {
		return nil, errors.New("list videos use case is required")
	}
	if deps.UploadVideoUC == nil {
		return nil, errors.New("upload video use case is required")
	}
	if deps.DeviceMonitor == nil {
		return nil, errors.New("MQTT device monitor is required")
	}

	app := &App{
		config:                deps.Config,
		appContext:            appContext,
		useCase:               deps.UseCase,
		acceleratorClient:     deps.AcceleratorClient,
		processorCapsUC:       deps.ProcessorCapsUC,
		getSystemInfoUC:       deps.GetSystemInfoUC,
		featureFlagRepo:       deps.FeatureFlagRepo,
		listInputsUC:          deps.ListInputsUC,
		evaluateFFUC:          deps.EvaluateFFUC,
		streamVideoUC:         deps.StreamVideoUC,
		listAvailableImagesUC: deps.ListAvailableImagesUC,
		uploadImageUC:         deps.UploadImageUC,
		listVideosUC:          deps.ListVideosUC,
		uploadVideoUC:         deps.UploadVideoUC,
		videoRepository:       deps.VideoRepository,
		deviceMonitor:         deps.DeviceMonitor,
	}

	for _, opt := range opts {
		opt(app)
	}

	return app, nil
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
		a.streamVideoUC,
		a.acceleratorClient,
	)

	connectrpc.RegisterConfigService(
		mux,
		connectrpc.ConfigHandlerDeps{
			FeatureFlagRepo: a.featureFlagRepo,
			ListInputsUC:    a.listInputsUC,
			EvaluateFFUC:    a.evaluateFFUC,
			GetSystemInfoUC: a.getSystemInfoUC,
			ConfigManager:   a.config,
			ProcessorCapsUC: a.processorCapsUC,
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

	connectrpc.RegisterWebRTCSignalingService(
		mux,
		a.acceleratorClient,
		a.interceptors...,
	)

	connectrpc.RegisterRoutesWithHandler(mux, rpcHandler, a.interceptors...)

	connectrpc.RegisterRemoteManagementService(
		mux,
		a.acceleratorClient,
		a.config,
		a.deviceMonitor,
		a.interceptors...,
	)

	transcoder := connectrpc.SetupVanguardTranscoder(&connectrpc.VanguardConfig{
		ImageProcessorHandler: rpcHandler,
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
	a.setupConnectRPCServices(mux)
	handler := a.makeTelemetryMiddleware(mux)

	errChan := make(chan error, 2)

	go func() {
		log.Info().
			Str("port", a.config.Server.HTTPPort).
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
