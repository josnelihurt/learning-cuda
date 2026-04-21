package processor

import (
	"context"
	"fmt"

	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/src/go_api/pkg/domain"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type GRPCProcessor struct {
	gateway *AcceleratorGateway
}

func NewGRPCProcessor(gateway *AcceleratorGateway) *GRPCProcessor {
	return &GRPCProcessor{
		gateway: gateway,
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

func (p *GRPCProcessor) domainToProtoFilters(filters []domain.FilterType) ([]pb.FilterType, bool, bool, error) {
	if len(filters) == 0 || (len(filters) == 1 && filters[0] == domain.FilterNone) {
		return nil, false, false, nil
	}

	protoFilters := make([]pb.FilterType, 0, len(filters))
	hasBlur := false
	hasModel := false
	for _, filter := range filters {
		switch filter {
		case domain.FilterNone:
			protoFilters = append(protoFilters, pb.FilterType_FILTER_TYPE_NONE)
		case domain.FilterGrayscale:
			protoFilters = append(protoFilters, pb.FilterType_FILTER_TYPE_GRAYSCALE)
		case domain.FilterBlur:
			protoFilters = append(protoFilters, pb.FilterType_FILTER_TYPE_BLUR)
			hasBlur = true
		case domain.FilterModel:
			protoFilters = append(protoFilters, pb.FilterType_FILTER_TYPE_MODEL_INFERENCE)
			hasModel = true
		default:
			return nil, false, false, fmt.Errorf("unsupported filter type: %s", filter)
		}
	}
	return protoFilters, hasBlur, hasModel, nil
}

func (p *GRPCProcessor) buildModelParams(hasModel bool, modelParams *domain.ModelInferenceParameters) *pb.ModelInferenceParameters {
	if !hasModel {
		return nil
	}
	result := &pb.ModelInferenceParameters{
		ModelId:             "yolov10n",
		ConfidenceThreshold: 0.5,
	}
	if modelParams != nil {
		if modelParams.ModelID != "" {
			result.ModelId = modelParams.ModelID
		}
		if modelParams.ConfidenceThreshold > 0 {
			result.ConfidenceThreshold = modelParams.ConfidenceThreshold
		}
	}
	return result
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

	protoFilters, hasBlur, hasModel, err := p.domainToProtoFilters(opts.Filters)
	if err != nil {
		return nil, err
	}
	if protoFilters == nil {
		return img, nil
	}

	finalBlurParams := p.buildBlurParams(hasBlur, opts.BlurParams)
	finalModelParams := p.buildModelParams(hasModel, opts.ModelParams)
	protoAccelerator := p.domainToProtoAccelerator(opts.Accelerator)
	protoGrayscaleType := p.domainToProtoGrayscaleType(opts.GrayscaleType)
	traceID, spanID, traceFlags := extractTraceContext(ctx, span)

	channels := int32(3)
	if img.Width > 0 && img.Height > 0 {
		if computed := int32(len(img.Data) / (img.Width * img.Height)); computed > 0 {
			channels = computed
		}
	}

	procReq := &pb.ProcessImageRequest{
		ApiVersion:    "2.0.0",
		ImageData:     img.Data,
		Width:         int32(img.Width),
		Height:        int32(img.Height),
		Channels:      channels,
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
	if finalModelParams != nil {
		procReq.ModelParams = finalModelParams
	}

	span.AddEvent("gRPC call started")
	resp, err := p.gateway.ProcessImage(ctx, procReq)
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
