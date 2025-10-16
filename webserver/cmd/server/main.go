package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/jrb/cuda-learning/webserver/pkg/app"
	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"github.com/jrb/cuda-learning/webserver/pkg/container"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/processor"
	"github.com/jrb/cuda-learning/webserver/pkg/telemetry"
)

func main() {
	ctx := context.Background()

	di, err := container.New(ctx)
	if err != nil {
		log.Fatalf("Failed to initialize container: %v", err)
	}
	defer di.Close(ctx)

	tracerProvider, err := telemetry.New(
		ctx,
		di.Config.IsObservabilityEnabled(ctx),
		di.Config.ObservabilityConfig,
	)
	if err != nil {
		log.Printf("Warning: Failed to initialize telemetry: %v", err)
	}

	defer func() {
		if tracerProvider != nil {
			shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
			defer cancel()
			if err := tracerProvider.Shutdown(shutdownCtx); err != nil {
				log.Printf("Error shutting down tracer provider: %v", err)
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
			log.Fatal(err)
		}
	case sig := <-sigChan:
		log.Printf("Received signal %v, shutting down gracefully...", sig)
	}
}
