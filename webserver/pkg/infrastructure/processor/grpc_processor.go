package processor

import (
	"context"
	"fmt"

	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type GRPCProcessor struct {
	client *GRPCClient
}

func NewGRPCProcessor(client *GRPCClient) *GRPCProcessor {
	return &GRPCProcessor{
		client: client,
	}
}

func (p *GRPCProcessor) domainToProtoBorderMode(mode domain.BorderMode) pb.BorderMode {
	switch mode {
	case domain.BorderModeClamp:
		return pb.BorderMode_BORDER_MODE_CLAMP
	case domain.BorderModeReflect:
		return pb.BorderMode_BORDER_MODE_REFLECT
	case domain.BorderModeWrap:
		return pb.BorderMode_BORDER_MODE_WRAP
	default:
		return pb.BorderMode_BORDER_MODE_REFLECT
	}
}

func (p *GRPCProcessor) domainToProtoFilters(filters []domain.FilterType) ([]pb.FilterType, bool, error) {
	if len(filters) == 0 || (len(filters) == 1 && filters[0] == domain.FilterNone) {
		return nil, false, nil
	}

	protoFilters := make([]pb.FilterType, 0, len(filters))
	hasBlur := false
	for _, filter := range filters {
		switch filter {
		case domain.FilterNone:
			protoFilters = append(protoFilters, pb.FilterType_FILTER_TYPE_NONE)
		case domain.FilterGrayscale:
			protoFilters = append(protoFilters, pb.FilterType_FILTER_TYPE_GRAYSCALE)
		case domain.FilterBlur:
			protoFilters = append(protoFilters, pb.FilterType_FILTER_TYPE_BLUR)
			hasBlur = true
		default:
			return nil, false, fmt.Errorf("unsupported filter type: %s", filter)
		}
	}
	return protoFilters, hasBlur, nil
}

func (p *GRPCProcessor) buildBlurParams(hasBlur bool, blurParams *domain.BlurParameters) *pb.GaussianBlurParameters {
	if !hasBlur {
		return nil
	}
	if blurParams != nil {
		return &pb.GaussianBlurParameters{
			KernelSize: blurParams.KernelSize,
			Sigma:      blurParams.Sigma,
			BorderMode: p.domainToProtoBorderMode(blurParams.BorderMode),
			Separable:  blurParams.Separable,
		}
	}
	return &pb.GaussianBlurParameters{
		KernelSize: 5,
		Sigma:      1.0,
		BorderMode: pb.BorderMode_BORDER_MODE_REFLECT,
		Separable:  true,
	}
}

func (p *GRPCProcessor) domainToProtoAccelerator(accelerator domain.AcceleratorType) pb.AcceleratorType {
	switch accelerator {
	case domain.AcceleratorGPU:
		return pb.AcceleratorType_ACCELERATOR_TYPE_CUDA
	case domain.AcceleratorCPU:
		return pb.AcceleratorType_ACCELERATOR_TYPE_CPU
	default:
		return pb.AcceleratorType_ACCELERATOR_TYPE_CUDA
	}
}

func (p *GRPCProcessor) domainToProtoGrayscaleType(grayscaleType domain.GrayscaleType) pb.GrayscaleType {
	switch grayscaleType {
	case domain.GrayscaleBT601:
		return pb.GrayscaleType_GRAYSCALE_TYPE_BT601
	case domain.GrayscaleBT709:
		return pb.GrayscaleType_GRAYSCALE_TYPE_BT709
	case domain.GrayscaleAverage:
		return pb.GrayscaleType_GRAYSCALE_TYPE_AVERAGE
	case domain.GrayscaleLightness:
		return pb.GrayscaleType_GRAYSCALE_TYPE_LIGHTNESS
	case domain.GrayscaleLuminosity:
		return pb.GrayscaleType_GRAYSCALE_TYPE_LUMINOSITY
	default:
		return pb.GrayscaleType_GRAYSCALE_TYPE_BT601
	}
}

func extractTraceContext(ctx context.Context, span trace.Span) (traceID, spanID string, traceFlags uint32) {
	spanContext := trace.SpanContextFromContext(ctx)
	if spanContext.IsValid() {
		traceID = spanContext.TraceID().String()
		spanID = spanContext.SpanID().String()
		traceFlags = uint32(spanContext.TraceFlags())
		span.SetAttributes(
			attribute.String("trace.id", traceID),
			attribute.String("span.id", spanID),
		)
	}
	return traceID, spanID, traceFlags
}

// ProcessImage mirrors the transformation logic of CppConnector.ProcessImage
// but sends the request over gRPC instead of calling the dynamic library.
func (p *GRPCProcessor) ProcessImage(
	ctx context.Context,
	img *domain.Image,
	opts domain.ProcessingOptions,
) (*domain.Image, error) {
	tracer := otel.Tracer("grpc-processor")
	ctx, span := tracer.Start(ctx, "GRPCProcessor.ProcessImage",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(
		attribute.String("accelerator", string(opts.Accelerator)),
		attribute.String("grayscale_type", string(opts.GrayscaleType)),
	)

	protoFilters, hasBlur, err := p.domainToProtoFilters(opts.Filters)
	if err != nil {
		return nil, err
	}
	if protoFilters == nil {
		return img, nil
	}

	finalBlurParams := p.buildBlurParams(hasBlur, opts.BlurParams)
	protoAccelerator := p.domainToProtoAccelerator(opts.Accelerator)
	protoGrayscaleType := p.domainToProtoGrayscaleType(opts.GrayscaleType)
	traceID, spanID, traceFlags := extractTraceContext(ctx, span)

	procReq := &pb.ProcessImageRequest{
		ApiVersion:    "2.0.0",
		ImageData:     img.Data,
		Width:         int32(img.Width),
		Height:        int32(img.Height),
		Channels:      int32(3),
		Filters:       protoFilters,
		Accelerator:   protoAccelerator,
		GrayscaleType: protoGrayscaleType,
		TraceId:       traceID,
		SpanId:        spanID,
		TraceFlags:    traceFlags,
	}
	if finalBlurParams != nil {
		procReq.BlurParams = finalBlurParams
	}

	span.AddEvent("gRPC call started")
	resp, err := p.client.ProcessImage(ctx, procReq)
	if err != nil {
		span.RecordError(err)
		return nil, fmt.Errorf("processing failed via gRPC: %w", err)
	}
	span.AddEvent("gRPC call completed")

	if resp.Code != 0 {
		err := fmt.Errorf("processing failed via gRPC: %s", resp.Message)
		span.RecordError(err)
		return nil, err
	}

	result := &domain.Image{
		Data:   resp.ImageData,
		Width:  int(resp.Width),
		Height: int(resp.Height),
		Format: img.Format,
	}

	return result, nil
}
