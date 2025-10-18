package featureflags

import (
	"context"

	flipt "go.flipt.io/flipt-client"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
)

type FliptClientProxy struct {
	client *flipt.Client
}

func NewFliptClient(client *flipt.Client) *FliptClientProxy {
	return &FliptClientProxy{client: client}
}

func (p *FliptClientProxy) EvaluateBoolean(ctx context.Context, req *flipt.EvaluationRequest) (*flipt.BooleanEvaluationResponse, error) {
	tracer := otel.Tracer("featureflags.FliptClient")
	ctx, span := tracer.Start(ctx, "EvaluateBoolean")
	defer span.End()

	span.SetAttributes(
		attribute.String("flipt.flag_key", req.FlagKey),
		attribute.String("flipt.entity_id", req.EntityID),
	)

	result, err := p.client.EvaluateBoolean(ctx, req)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, "Flipt evaluation failed")
		return nil, err
	}

	span.SetAttributes(attribute.Bool("flipt.result", result.Enabled))
	span.SetStatus(codes.Ok, "Flipt evaluation successful")
	return result, nil
}

func (p *FliptClientProxy) Close(ctx context.Context) error {
	tracer := otel.Tracer("featureflags.FliptClient")
	ctx, span := tracer.Start(ctx, "Close")
	defer span.End()

	err := p.client.Close(ctx)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, "Failed to close Flipt client")
		return err
	}

	span.SetStatus(codes.Ok, "Flipt client closed successfully")
	return nil
}

func (p *FliptClientProxy) EvaluateString(ctx context.Context, req *flipt.EvaluationRequest) (*flipt.VariantEvaluationResponse, error) {
	tracer := otel.Tracer("featureflags.FliptClient")
	ctx, span := tracer.Start(ctx, "EvaluateString")
	defer span.End()

	result, err := p.client.EvaluateVariant(ctx, req)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, "Flipt evaluation failed")
		return nil, err
	}

	span.SetAttributes(attribute.String("flipt.result", result.VariantKey))
	span.SetStatus(codes.Ok, "Flipt evaluation successful")
	return result, nil
}
