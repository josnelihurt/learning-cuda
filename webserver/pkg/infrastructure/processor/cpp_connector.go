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

type CppConnector struct {
	loader *loader.Loader
}

func New(libraryPath string) (*CppConnector, error) {
	l, err := loader.NewLoader(libraryPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load processor library: %w", err)
	}

	initReq := &pb.InitRequest{
		ApiVersion:   loader.CurrentAPIVersion,
		CudaDeviceId: 0,
	}

	initResp, err := l.Init(initReq)
	if err != nil {
		l.Cleanup()
		return nil, fmt.Errorf("initialization failed: %w", err)
	}

	if initResp.Code != 0 {
		l.Cleanup()
		return nil, fmt.Errorf("init failed: %s", initResp.Message)
	}

	return &CppConnector{loader: l}, nil
}

//nolint:gocyclo // Complex processing logic that needs to handle multiple filter types
func (c *CppConnector) ProcessImage(ctx context.Context, img *domain.Image, filters []domain.FilterType, accelerator domain.AcceleratorType, grayscaleType domain.GrayscaleType) (*domain.Image, error) {
	tracer := otel.Tracer("cpp-connector")
	ctx, span := tracer.Start(ctx, "CppConnector.ProcessImage",
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
	for _, filter := range filters {
		switch filter {
		case domain.FilterNone:
			protoFilters = append(protoFilters, pb.FilterType_FILTER_TYPE_NONE)
		case domain.FilterGrayscale:
			protoFilters = append(protoFilters, pb.FilterType_FILTER_TYPE_GRAYSCALE)
		default:
			return nil, fmt.Errorf("unsupported filter type: %s", filter)
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
		Channels:      int32(4),
		Filters:       protoFilters,
		Accelerator:   protoAccelerator,
		GrayscaleType: protoGrayscaleType,
		TraceId:       traceID,
		SpanId:        spanID,
		TraceFlags:    traceFlags,
	}

	span.AddEvent("Dynamic library call started")
	procResp, err := c.loader.ProcessImage(procReq)
	if err != nil {
		span.RecordError(err)
		return nil, fmt.Errorf("processing failed: %w", err)
	}
	span.AddEvent("Dynamic library call completed")

	if procResp.Code != 0 {
		err := fmt.Errorf("processing failed: %s", procResp.Message)
		span.RecordError(err)
		return nil, err
	}

	result := &domain.Image{
		Data:   procResp.ImageData,
		Width:  int(procResp.Width),
		Height: int(procResp.Height),
		Format: img.Format,
	}

	return result, nil
}

func (c *CppConnector) Close() {
	if c.loader != nil {
		c.loader.Cleanup()
	}
}
