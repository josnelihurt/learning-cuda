package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/jrb/cuda-learning/webserver/internal/app"
	"github.com/jrb/cuda-learning/webserver/internal/application"
	"github.com/jrb/cuda-learning/webserver/internal/infrastructure/processor"
	"github.com/jrb/cuda-learning/webserver/internal/telemetry"
)

func main() {
	config := app.LoadConfig()
	
	tracerProvider, err := telemetry.NewTracerProvider(
		config.Observability.ServiceName,
		config.Observability.ServiceVersion,
		config.Observability.OtelCollectorEndpoint,
		config.Observability.TraceSamplingRate,
		config.IsFeatureEnabled("enable_observability"),
	)
	if err != nil {
		log.Printf("Warning: Failed to initialize telemetry: %v", err)
	}
	
	defer func() {
		if tracerProvider != nil {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if err := tracerProvider.Shutdown(ctx); err != nil {
				log.Printf("Error shutting down tracer provider: %v", err)
			}
		}
	}()
	
	cppConnector := processor.NewCppConnector()
	processImageUseCase := application.NewProcessImageUseCase(cppConnector)
	
	server := app.New(config, processImageUseCase)
	
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

