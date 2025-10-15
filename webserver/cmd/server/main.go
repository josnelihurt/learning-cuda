package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/jrb/cuda-learning/webserver/internal/app"
	"github.com/jrb/cuda-learning/webserver/internal/application"
	"github.com/jrb/cuda-learning/webserver/internal/config"
	httpinfra "github.com/jrb/cuda-learning/webserver/internal/infrastructure/http"
	"github.com/jrb/cuda-learning/webserver/internal/infrastructure/processor"
	"github.com/jrb/cuda-learning/webserver/internal/telemetry"
	"go.flipt.io/flipt-client"
)

func main() {
	ctx := context.Background()
	cfg := config.New()

	fliptClient, err := flipt.NewClient(context.Background(), flipt.WithURL(cfg.FliptConfig.URL), flipt.WithNamespace(cfg.FliptConfig.Namespace))
	if err != nil {
		log.Fatalf("Failed to create Flipt client: %v", err)
	}
	fliptClientProxy := config.NewFliptClient(fliptClient)
	httpClientProxy := httpinfra.New(&http.Client{
		Timeout: cfg.HttpClientTimeout,
	})
	fliptWriter := config.NewFliptWriter(cfg.FliptConfig.URL, cfg.FliptConfig.Namespace, httpClientProxy)
	featureFlagsManager := config.NewFeatureFlagManager(fliptClientProxy, fliptWriter)
	config.WithFeatureFlagManager(featureFlagsManager)

	tracerProvider, err := telemetry.New(ctx, featureFlagsManager.IsObservabilityEnabled(ctx), cfg.ObservabilityConfig)
	if err != nil {
		log.Printf("Warning: Failed to initialize telemetry: %v", err)
	}

	defer func() {
		cfg.Close()

		if tracerProvider != nil {
			ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
			defer cancel()
			if err := tracerProvider.Shutdown(ctx); err != nil {
				log.Printf("Error shutting down tracer provider: %v", err)
			}
		}
	}()

	cppConnector := processor.New()
	processImageUseCase := application.NewProcessImageUseCase(cppConnector)

	server := app.New(
		ctx,
		app.WithConfig(cfg),
		app.WithUseCase(processImageUseCase),
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
