package application

import (
	"context"

	"github.com/jrb/cuda-learning/webserver/pkg/config"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type GetStreamConfigUseCase struct {
	evaluateFFUseCase *EvaluateFeatureFlagUseCase
	defaultConfig     config.StreamConfig
}

func NewGetStreamConfigUseCase(
	evaluateUC *EvaluateFeatureFlagUseCase,
	defaultCfg config.StreamConfig,
) *GetStreamConfigUseCase {
	return &GetStreamConfigUseCase{
		evaluateFFUseCase: evaluateUC,
		defaultConfig:     defaultCfg,
	}
}

func (uc *GetStreamConfigUseCase) Execute(ctx context.Context) (*config.StreamConfig, error) {
	tracer := otel.Tracer("get-stream-config")
	ctx, span := tracer.Start(ctx, "GetStreamConfig",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	transportFormat, err := uc.evaluateFFUseCase.EvaluateVariant(
		ctx,
		"ws_transport_format",
		"stream_transport_format",
		uc.defaultConfig.TransportFormat,
	)
	if err != nil {
		span.RecordError(err)
		return nil, err
	}

	span.SetAttributes(
		attribute.String("config.transport_format", transportFormat),
		attribute.String("config.websocket_endpoint", uc.defaultConfig.WebsocketEndpoint),
	)

	return &config.StreamConfig{
		TransportFormat:   transportFormat,
		WebsocketEndpoint: uc.defaultConfig.WebsocketEndpoint,
	}, nil
}
