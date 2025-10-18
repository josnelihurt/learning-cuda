package telemetry

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/jrb/cuda-learning/webserver/pkg/config"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type TracerProvider struct {
	provider *sdktrace.TracerProvider
	enabled  bool
}

func New(ctx context.Context, enabled bool, config config.ObservabilityConfig) (*TracerProvider, error) {
	if !enabled {
		log.Println("Observability disabled by feature flag")
		return &TracerProvider{enabled: false}, nil
	}

	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	res, err := resource.New(ctx,
		resource.WithAttributes(
			attribute.String("service.name", config.ServiceName),
			attribute.String("service.version", config.ServiceVersion),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create resource: %w", err)
	}

	conn, err := grpc.NewClient(config.OtelCollectorEndpoint,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to OpenTelemetry collector at %s: %w", config.OtelCollectorEndpoint, err)
	}

	exporter, err := otlptracegrpc.New(ctx, otlptracegrpc.WithGRPCConn(conn))
	if err != nil {
		return nil, fmt.Errorf("failed to create OTLP trace exporter: %w", err)
	}

	samplingRate := config.TraceSamplingRate
	var sampler sdktrace.Sampler
	switch {
	case samplingRate >= 1.0:
		sampler = sdktrace.AlwaysSample()
	case samplingRate <= 0.0:
		sampler = sdktrace.NeverSample()
	default:
		sampler = sdktrace.TraceIDRatioBased(samplingRate)
	}

	provider := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sampler),
	)

	otel.SetTracerProvider(provider)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	log.Printf("OpenTelemetry tracer initialized (endpoint: %s, sampling: %.2f)", config.OtelCollectorEndpoint, config.TraceSamplingRate)

	return &TracerProvider{
		provider: provider,
		enabled:  true,
	}, nil
}

func (tp *TracerProvider) Shutdown(ctx context.Context) error {
	if !tp.enabled || tp.provider == nil {
		return nil
	}

	log.Println("Shutting down OpenTelemetry tracer provider...")
	if err := tp.provider.Shutdown(ctx); err != nil {
		return fmt.Errorf("failed to shutdown tracer provider: %w", err)
	}
	log.Println("OpenTelemetry tracer provider shutdown complete")
	return nil
}

func (tp *TracerProvider) IsEnabled() bool {
	return tp.enabled
}
