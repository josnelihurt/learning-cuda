package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/jrb/cuda-learning/webserver/pkg/app"
	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"github.com/jrb/cuda-learning/webserver/pkg/container"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/logger"
	"github.com/jrb/cuda-learning/webserver/pkg/telemetry"
)

func main() {
	ctx := context.Background()

	di, err := container.New(ctx)
	if err != nil {
		logger.Global().Fatal().Err(err).Msg("Failed to initialize container")
	}

	log := logger.Global()

	tracerProvider, err := telemetry.New(
		ctx,
		di.Config.IsObservabilityEnabled(ctx),
		di.Config.Observability,
	)
	if err != nil {
		log.Warn().Err(err).Msg("Failed to initialize telemetry")
	}

	processImageUseCase := application.NewProcessImageUseCase(di.CppConnector)

	server := app.New(
		ctx,
		app.WithConfig(di.Config),
		app.WithUseCase(processImageUseCase),
		app.WithGetStreamConfigUseCase(di.GetStreamConfigUseCase),
		app.WithGetSystemInfoUseCase(di.GetSystemInfoUseCase),
		app.WithSyncFlagsUseCase(di.SyncFeatureFlagsUseCase),
		app.WithListInputsUseCase(di.ListInputsUseCase),
		app.WithEvaluateFFUseCase(di.EvaluateFeatureFlagUseCase),
		app.WithListAvailableImagesUseCase(di.ListAvailableImagesUseCase),
		app.WithUploadImageUseCase(di.UploadImageUseCase),
		app.WithListVideosUseCase(di.ListVideosUseCase),
		app.WithUploadVideoUseCase(di.UploadVideoUseCase),
		app.WithVideoRepository(di.VideoRepository),
		app.WithProcessorRegistry(di.ProcessorRegistry),
		app.WithProcessorLoader(&di.ProcessorLoader),
		app.WithLoaderMutex(di.LoaderMutex),
	)

	errChan := make(chan error, 1)
	go func() {
		errChan <- server.Run()
	}()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	select {
	case err := <-errChan:
		if err != nil {
			log.Error().Err(err).Msg("Server error")
			di.Close(ctx)
			if tracerProvider != nil {
				shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
				if shutdownErr := tracerProvider.Shutdown(shutdownCtx); shutdownErr != nil {
					log.Error().Err(shutdownErr).Msg("Error shutting down tracer provider")
				}
				cancel()
			}
			os.Exit(1)
		}
	case sig := <-sigChan:
		log.Info().Str("signal", sig.String()).Msg("Received signal, shutting down gracefully")
	}

	di.Close(ctx)
	if tracerProvider != nil {
		shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		if err := tracerProvider.Shutdown(shutdownCtx); err != nil {
			log.Error().Err(err).Msg("Error shutting down tracer provider")
		}
	}
}
