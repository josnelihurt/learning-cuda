package adapters

import (
	"log"
	"strconv"
	"strings"

	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/webserver/pkg/domain"
)

type ProtobufAdapter struct{}

func NewProtobufAdapter() *ProtobufAdapter {
	return &ProtobufAdapter{}
}

func (a *ProtobufAdapter) ExtractProcessingOptions(req *pb.ProcessImageRequest) domain.ProcessingOptions {
	if req == nil {
		return domain.ProcessingOptions{
			GrayscaleType: domain.GrayscaleBT601,
			Accelerator:   domain.AcceleratorGPU,
		}
	}

	filters, grayscale, blurParams := a.resolveProtoProcessingFields(req)

	return domain.ProcessingOptions{
		Filters:       a.ToFilters(filters),
		Accelerator:   a.ToAccelerator(req.Accelerator),
		GrayscaleType: a.ToGrayscaleType(grayscale),
		BlurParams:    a.ToBlurParameters(blurParams),
	}
}

func (a *ProtobufAdapter) resolveProtoProcessingFields(req *pb.ProcessImageRequest) ([]pb.FilterType, pb.GrayscaleType, *pb.GaussianBlurParameters) {
	filters := req.Filters
	grayscale := req.GrayscaleType
	blurParams := req.BlurParams

	if len(req.GenericFilters) > 0 {
		genericFilters, genericGrayscale, genericBlur := a.genericSelectionsToProto(req.GenericFilters, grayscale, blurParams)
		if len(genericFilters) > 0 {
			filters = genericFilters
		}
		grayscale = genericGrayscale
		blurParams = genericBlur
	}

	return filters, grayscale, blurParams
}

func (a *ProtobufAdapter) ToFilters(pbFilters []pb.FilterType) []domain.FilterType {
	filters := make([]domain.FilterType, 0, len(pbFilters))
	for _, f := range pbFilters {
		switch f {
		case pb.FilterType_FILTER_TYPE_GRAYSCALE:
			filters = append(filters, domain.FilterGrayscale)
		case pb.FilterType_FILTER_TYPE_BLUR:
			filters = append(filters, domain.FilterBlur)
		case pb.FilterType_FILTER_TYPE_NONE:
			filters = append(filters, domain.FilterNone)
		case pb.FilterType_FILTER_TYPE_UNSPECIFIED:
			// Log warning for unspecified filter type
			log.Printf("Warning: received unspecified filter type, skipping")
		default:
			// Log warning for unknown filter type
			log.Printf("Warning: received unknown filter type %v, skipping", f)
		}
	}
	return filters
}

func (a *ProtobufAdapter) ToAccelerator(pbAccel pb.AcceleratorType) domain.AcceleratorType {
	switch pbAccel {
	case pb.AcceleratorType_ACCELERATOR_TYPE_CUDA:
		return domain.AcceleratorGPU
	case pb.AcceleratorType_ACCELERATOR_TYPE_CPU:
		return domain.AcceleratorCPU
	case pb.AcceleratorType_ACCELERATOR_TYPE_UNSPECIFIED:
		// Log warning for unspecified accelerator type
		log.Printf("Warning: received unspecified accelerator type, defaulting to GPU")
		return domain.AcceleratorGPU
	case pb.AcceleratorType_ACCELERATOR_TYPE_OPENCL:
		// Log warning for unsupported accelerator type
		log.Printf("Warning: received unsupported accelerator type OPENCL, defaulting to GPU")
		return domain.AcceleratorGPU
	default:
		// Log warning for unknown accelerator type
		log.Printf("Warning: received unknown accelerator type %v, defaulting to GPU", pbAccel)
		return domain.AcceleratorGPU
	}
}

func (a *ProtobufAdapter) ToGrayscaleType(pbType pb.GrayscaleType) domain.GrayscaleType {
	switch pbType {
	case pb.GrayscaleType_GRAYSCALE_TYPE_BT601:
		return domain.GrayscaleBT601
	case pb.GrayscaleType_GRAYSCALE_TYPE_BT709:
		return domain.GrayscaleBT709
	case pb.GrayscaleType_GRAYSCALE_TYPE_AVERAGE:
		return domain.GrayscaleAverage
	case pb.GrayscaleType_GRAYSCALE_TYPE_LIGHTNESS:
		return domain.GrayscaleLightness
	case pb.GrayscaleType_GRAYSCALE_TYPE_LUMINOSITY:
		return domain.GrayscaleLuminosity
	case pb.GrayscaleType_GRAYSCALE_TYPE_UNSPECIFIED:
		// Log warning for unspecified grayscale type
		log.Printf("Warning: received unspecified grayscale type, defaulting to BT601")
		return domain.GrayscaleBT601
	default:
		// Log warning for unknown grayscale type
		log.Printf("Warning: received unknown grayscale type %v, defaulting to BT601", pbType)
		return domain.GrayscaleBT601
	}
}

func (a *ProtobufAdapter) ToBorderMode(pbMode pb.BorderMode) domain.BorderMode {
	switch pbMode {
	case pb.BorderMode_BORDER_MODE_CLAMP:
		return domain.BorderModeClamp
	case pb.BorderMode_BORDER_MODE_REFLECT:
		return domain.BorderModeReflect
	case pb.BorderMode_BORDER_MODE_WRAP:
		return domain.BorderModeWrap
	case pb.BorderMode_BORDER_MODE_UNSPECIFIED:
		// Log warning for unspecified border mode
		log.Printf("Warning: received unspecified border mode, defaulting to REFLECT")
		return domain.BorderModeReflect
	default:
		// Log warning for unknown border mode
		log.Printf("Warning: received unknown border mode %v, defaulting to REFLECT", pbMode)
		return domain.BorderModeReflect
	}
}

func (a *ProtobufAdapter) ToBlurParameters(pbBlur *pb.GaussianBlurParameters) *domain.BlurParameters {
	if pbBlur == nil {
		return nil
	}

	return &domain.BlurParameters{
		KernelSize: pbBlur.KernelSize,
		Sigma:      pbBlur.Sigma,
		BorderMode: a.ToBorderMode(pbBlur.BorderMode),
		Separable:  pbBlur.Separable,
	}
}

func (a *ProtobufAdapter) ToProtobufBorderMode(mode domain.BorderMode) pb.BorderMode {
	switch mode {
	case domain.BorderModeClamp:
		return pb.BorderMode_BORDER_MODE_CLAMP
	case domain.BorderModeReflect:
		return pb.BorderMode_BORDER_MODE_REFLECT
	case domain.BorderModeWrap:
		return pb.BorderMode_BORDER_MODE_WRAP
	default:
		log.Printf("Warning: unknown border mode %v, defaulting to REFLECT", mode)
		return pb.BorderMode_BORDER_MODE_REFLECT
	}
}

func (a *ProtobufAdapter) ToProtobufBlurParameters(blur *domain.BlurParameters) *pb.GaussianBlurParameters {
	if blur == nil {
		return nil
	}

	return &pb.GaussianBlurParameters{
		KernelSize: blur.KernelSize,
		Sigma:      blur.Sigma,
		BorderMode: a.ToProtobufBorderMode(blur.BorderMode),
		Separable:  blur.Separable,
	}
}

func (a *ProtobufAdapter) ToDomainImage(req *pb.ProcessImageRequest) *domain.Image {
	return &domain.Image{
		Data:   req.ImageData,
		Width:  int(req.Width),
		Height: int(req.Height),
		Format: "raw",
	}
}

func (a *ProtobufAdapter) ToProtobufResponse(img *domain.Image) *pb.ProcessImageResponse {
	return &pb.ProcessImageResponse{
		Code:      0,
		Message:   "success",
		ImageData: img.Data,
		Width:     int32(img.Width),
		Height:    int32(img.Height),
		Channels:  int32(len(img.Data) / (img.Width * img.Height)),
	}
}

func (a *ProtobufAdapter) genericSelectionsToProto(
	selections []*pb.GenericFilterSelection,
	fallbackGrayscale pb.GrayscaleType,
	fallbackBlur *pb.GaussianBlurParameters,
) ([]pb.FilterType, pb.GrayscaleType, *pb.GaussianBlurParameters) {
	filters := make([]pb.FilterType, 0, len(selections))
	grayscale := fallbackGrayscale
	blur := fallbackBlur

	for _, selection := range selections {
		if selection == nil {
			continue
		}

		switch strings.ToLower(strings.TrimSpace(selection.FilterId)) {
		case "", "none":
			filters = append(filters, pb.FilterType_FILTER_TYPE_NONE)
		case "grayscale":
			filters = append(filters, pb.FilterType_FILTER_TYPE_GRAYSCALE)
			grayscale = a.grayscaleFromGeneric(selection.Parameters, grayscale)
		case "blur":
			filters = append(filters, pb.FilterType_FILTER_TYPE_BLUR)
			blur = a.blurFromGeneric(selection.Parameters, blur)
		default:
			log.Printf("Warning: received unknown generic filter %s, skipping", selection.FilterId)
		}
	}

	if len(filters) == 0 {
		return nil, fallbackGrayscale, fallbackBlur
	}

	return filters, grayscale, blur
}

func (a *ProtobufAdapter) grayscaleFromGeneric(params []*pb.GenericFilterParameterSelection, fallback pb.GrayscaleType) pb.GrayscaleType {
	for _, param := range params {
		if param == nil {
			continue
		}
		if strings.EqualFold(param.ParameterId, "algorithm") {
			if value, ok := firstGenericValue(param); ok {
				return a.mapStringToGrayscaleType(value)
			}
		}
	}

	if fallback == pb.GrayscaleType_GRAYSCALE_TYPE_UNSPECIFIED {
		return pb.GrayscaleType_GRAYSCALE_TYPE_BT601
	}
	return fallback
}

func (a *ProtobufAdapter) blurFromGeneric(params []*pb.GenericFilterParameterSelection, fallback *pb.GaussianBlurParameters) *pb.GaussianBlurParameters {
	result := a.cloneBlurParams(fallback)

	for _, param := range params {
		if param == nil {
			continue
		}

		value, ok := firstGenericValue(param)
		if !ok {
			continue
		}

		switch strings.ToLower(param.ParameterId) {
		case "kernel_size":
			if parsed, err := strconv.ParseInt(value, 10, 32); err == nil {
				if parsed < 1 {
					parsed = 1
				}
				if parsed%2 == 0 {
					parsed++
				}
				const maxInt32 = 2147483647
				if parsed > maxInt32 {
					parsed = maxInt32
				}
				result.KernelSize = int32(parsed)
			}
		case "sigma":
			if parsed, err := strconv.ParseFloat(value, 32); err == nil && parsed >= 0 {
				result.Sigma = float32(parsed)
			}
		case "border_mode":
			result.BorderMode = mapStringToBorderMode(value)
		case "separable":
			result.Separable = parseBool(value)
		}
	}

	return result
}

func (a *ProtobufAdapter) mapStringToGrayscaleType(value string) pb.GrayscaleType {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "bt709":
		return pb.GrayscaleType_GRAYSCALE_TYPE_BT709
	case "average":
		return pb.GrayscaleType_GRAYSCALE_TYPE_AVERAGE
	case "lightness":
		return pb.GrayscaleType_GRAYSCALE_TYPE_LIGHTNESS
	case "luminosity":
		return pb.GrayscaleType_GRAYSCALE_TYPE_LUMINOSITY
	case "bt601":
		return pb.GrayscaleType_GRAYSCALE_TYPE_BT601
	default:
		return pb.GrayscaleType_GRAYSCALE_TYPE_BT601
	}
}

func (a *ProtobufAdapter) cloneBlurParams(source *pb.GaussianBlurParameters) *pb.GaussianBlurParameters {
	if source == nil {
		return &pb.GaussianBlurParameters{
			KernelSize: 5,
			Sigma:      1.0,
			BorderMode: pb.BorderMode_BORDER_MODE_REFLECT,
			Separable:  true,
		}
	}

	return &pb.GaussianBlurParameters{
		KernelSize: source.KernelSize,
		Sigma:      source.Sigma,
		BorderMode: source.BorderMode,
		Separable:  source.Separable,
	}
}

func firstGenericValue(selection *pb.GenericFilterParameterSelection) (string, bool) {
	if selection == nil {
		return "", false
	}
	for _, value := range selection.Values {
		if value != "" {
			return value, true
		}
	}
	return "", false
}

func mapStringToBorderMode(value string) pb.BorderMode {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "clamp":
		return pb.BorderMode_BORDER_MODE_CLAMP
	case "wrap":
		return pb.BorderMode_BORDER_MODE_WRAP
	case "reflect":
		return pb.BorderMode_BORDER_MODE_REFLECT
	default:
		return pb.BorderMode_BORDER_MODE_REFLECT
	}
}

func parseBool(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "on", "yes":
		return true
	default:
		return false
	}
}
