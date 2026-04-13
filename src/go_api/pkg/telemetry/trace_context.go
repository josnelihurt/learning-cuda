package telemetry

import (
	"context"
	"net/http"

	pb "github.com/jrb/cuda-learning/proto/gen"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
)

// ExtractFromHTTPHeaders extracts trace context from HTTP headers
func ExtractFromHTTPHeaders(ctx context.Context, headers http.Header) context.Context {
	return otel.GetTextMapPropagator().Extract(ctx, propagation.HeaderCarrier(headers))
}

// ExtractFromProtobuf extracts trace context from protobuf TraceContext message
func ExtractFromProtobuf(ctx context.Context, traceCtx *pb.TraceContext) context.Context {
	if traceCtx == nil || traceCtx.Traceparent == "" {
		return ctx
	}

	carrier := propagation.MapCarrier{
		"traceparent": traceCtx.Traceparent,
	}
	if traceCtx.Tracestate != "" {
		carrier["tracestate"] = traceCtx.Tracestate
	}

	return otel.GetTextMapPropagator().Extract(ctx, carrier)
}

// InjectToProtobuf injects current trace context into protobuf TraceContext message
func InjectToProtobuf(ctx context.Context) *pb.TraceContext {
	carrier := propagation.MapCarrier{}
	otel.GetTextMapPropagator().Inject(ctx, carrier)

	return &pb.TraceContext{
		Traceparent: carrier["traceparent"],
		Tracestate:  carrier["tracestate"],
	}
}
