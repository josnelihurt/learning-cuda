package processor

import (
	"context"
	"fmt"

	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/processor/loader"
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

// ProcessImage mirrors the transformation logic of CppConnector.ProcessImage
// but sends the request over gRPC instead of calling the dynamic library.
func (p *GRPCProcessor) ProcessImage(
	ctx context.Context,
	img *domain.Image,
	filters []domain.FilterType,
	accelerator domain.AcceleratorType,
	grayscaleType domain.GrayscaleType,
	blurParams *domain.BlurParameters,
) (*domain.Image, error) {
	tracer := otel.Tracer("grpc-processor")
	ctx, span := tracer.Start(ctx, "GRPCProcessor.ProcessImage",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(
		attribute.String("accelerator", string(accelerator)),
		attribute.String("grayscale_type", string(grayscaleType)),
	)

	if len(filters) == 0 || (len(filters) == 1 && filters[0] == domain.FilterNone) {
		return img, nil
	}

	var protoFilters []pb.FilterType
	var finalBlurParams *pb.GaussianBlurParameters

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
			return nil, fmt.Errorf("unsupported filter type: %s", filter)
		}
	}

	if hasBlur {
		if blurParams != nil {
			finalBlurParams = &pb.GaussianBlurParameters{
				KernelSize: blurParams.KernelSize,
				Sigma:      blurParams.Sigma,
				BorderMode: p.domainToProtoBorderMode(blurParams.BorderMode),
				Separable:  blurParams.Separable,
			}
		} else {
			finalBlurParams = &pb.GaussianBlurParameters{
				KernelSize: 5,
				Sigma:      1.0,
				BorderMode: pb.BorderMode_BORDER_MODE_REFLECT,
				Separable:  true,
			}
		}
	}

	var protoAccelerator pb.AcceleratorType
	switch accelerator {
	case domain.AcceleratorGPU:
		protoAccelerator = pb.AcceleratorType_ACCELERATOR_TYPE_CUDA
	case domain.AcceleratorCPU:
		protoAccelerator = pb.AcceleratorType_ACCELERATOR_TYPE_CPU
	default:
		protoAccelerator = pb.AcceleratorType_ACCELERATOR_TYPE_CUDA
	}

	var protoGrayscaleType pb.GrayscaleType
	switch grayscaleType {
	case domain.GrayscaleBT601:
		protoGrayscaleType = pb.GrayscaleType_GRAYSCALE_TYPE_BT601
	case domain.GrayscaleBT709:
		protoGrayscaleType = pb.GrayscaleType_GRAYSCALE_TYPE_BT709
	case domain.GrayscaleAverage:
		protoGrayscaleType = pb.GrayscaleType_GRAYSCALE_TYPE_AVERAGE
	case domain.GrayscaleLightness:
		protoGrayscaleType = pb.GrayscaleType_GRAYSCALE_TYPE_LIGHTNESS
	case domain.GrayscaleLuminosity:
		protoGrayscaleType = pb.GrayscaleType_GRAYSCALE_TYPE_LUMINOSITY
	default:
		protoGrayscaleType = pb.GrayscaleType_GRAYSCALE_TYPE_BT601
	}

	spanContext := trace.SpanContextFromContext(ctx)
	var traceID, spanID string
	var traceFlags uint32
	if spanContext.IsValid() {
		traceID = spanContext.TraceID().String()
		spanID = spanContext.SpanID().String()
		traceFlags = uint32(spanContext.TraceFlags())
		span.SetAttributes(
			attribute.String("trace.id", traceID),
			attribute.String("span.id", spanID),
		)
	}

	procReq := &pb.ProcessImageRequest{
		ApiVersion:    loader.CurrentAPIVersion,
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
