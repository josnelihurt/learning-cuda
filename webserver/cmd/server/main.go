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
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/processor"
	"github.com/jrb/cuda-learning/webserver/pkg/telemetry"
)

func main() {
	ctx := context.Background()

	di, err := container.New(ctx)
	if err != nil {
		logger.Global().Fatal().Err(err).Msg("Failed to initialize container")
	}
	defer di.Close(ctx)

	log := logger.Global()

	tracerProvider, err := telemetry.New(
		ctx,
		di.Config.IsObservabilityEnabled(ctx),
		di.Config.ObservabilityConfig,
	)
	if err != nil {
		log.Warn().Err(err).Msg("Failed to initialize telemetry")
	}

	defer func() {
		if tracerProvider != nil {
			shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
			defer cancel()
			if err := tracerProvider.Shutdown(shutdownCtx); err != nil {
				log.Error().Err(err).Msg("Error shutting down tracer provider")
			}
		}
	}()

	cppConnector := processor.New()
	processImageUseCase := application.NewProcessImageUseCase(cppConnector)

	server := app.New(
		ctx,
		app.WithConfig(di.Config),
		app.WithUseCase(processImageUseCase),
		app.WithGetStreamConfigUseCase(di.GetStreamConfigUseCase),
		app.WithSyncFlagsUseCase(di.SyncFeatureFlagsUseCase),
		app.WithListInputsUseCase(di.ListInputsUseCase),
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
			log.Fatal().Err(err).Msg("Server error")
		}
	case sig := <-sigChan:
		log.Info().Str("signal", sig.String()).Msg("Received signal, shutting down gracefully")
	}
}
