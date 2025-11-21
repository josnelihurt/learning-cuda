package adapters

import (
	"strings"

	pb "github.com/jrb/cuda-learning/proto/gen"
)

const (
	paramTypeSelect   = "select"
	paramTypeCheckbox = "checkbox"
	paramTypeText     = "text"
	filterIDBlur      = "blur"
)

// FilterCodec provides bidirectional conversion between FilterDefinition and GenericFilterDefinition
type FilterCodec struct{}

// NewFilterCodec creates a new FilterCodec instance
func NewFilterCodec() *FilterCodec {
	return &FilterCodec{}
}

// ToGenericFilterDefinition converts a FilterDefinition to a GenericFilterDefinition
func (c *FilterCodec) ToGenericFilterDefinition(def *pb.FilterDefinition) *pb.GenericFilterDefinition {
	if def == nil {
		return nil
	}

	parameters := make([]*pb.GenericFilterParameter, 0, len(def.Parameters))
	for _, param := range def.Parameters {
		if param == nil {
			continue
		}
		genericParam := c.ToGenericFilterParameter(def.Id, param)
		if genericParam != nil {
			parameters = append(parameters, genericParam)
		}
	}

	return &pb.GenericFilterDefinition{
		Id:                    def.Id,
		Name:                  def.Name,
		Parameters:            parameters,
		SupportedAccelerators: def.SupportedAccelerators,
	}
}

// ToFilterDefinition converts a GenericFilterDefinition to a FilterDefinition
func (c *FilterCodec) ToFilterDefinition(genericDef *pb.GenericFilterDefinition) *pb.FilterDefinition {
	if genericDef == nil {
		return nil
	}

	parameters := make([]*pb.FilterParameter, 0, len(genericDef.Parameters))
	for _, genericParam := range genericDef.Parameters {
		if genericParam == nil {
			continue
		}
		param := c.ToFilterParameter(genericParam)
		if param != nil {
			parameters = append(parameters, param)
		}
	}

	return &pb.FilterDefinition{
		Id:                    genericDef.Id,
		Name:                  genericDef.Name,
		Parameters:            parameters,
		SupportedAccelerators: genericDef.SupportedAccelerators,
	}
}

// ToGenericFilterParameter converts a FilterParameter to a GenericFilterParameter
func (c *FilterCodec) ToGenericFilterParameter(filterID string, param *pb.FilterParameter) *pb.GenericFilterParameter {
	if param == nil {
		return nil
	}

	options := make([]*pb.GenericFilterParameterOption, 0, len(param.Options))
	for _, option := range param.Options {
		options = append(options, &pb.GenericFilterParameterOption{
			Value: option,
			Label: c.FormatParameterLabel(option),
		})
	}

	return &pb.GenericFilterParameter{
		Id:           param.Id,
		Name:         param.Name,
		Type:         c.MapParameterTypeToGeneric(param.Type),
		Options:      options,
		DefaultValue: param.DefaultValue,
		Metadata:     c.BuildParameterMetadata(filterID, param),
	}
}

// ToFilterParameter converts a GenericFilterParameter to a FilterParameter
func (c *FilterCodec) ToFilterParameter(genericParam *pb.GenericFilterParameter) *pb.FilterParameter {
	if genericParam == nil {
		return nil
	}

	// Convert GenericFilterParameterOption slice to string slice (option values)
	options := make([]string, 0, len(genericParam.Options))
	for _, option := range genericParam.Options {
		if option != nil && option.Value != "" {
			options = append(options, option.Value)
		}
	}

	// Convert GenericFilterParameterType enum to string
	paramType := c.MapGenericParameterTypeToString(genericParam.Type)

	return &pb.FilterParameter{
		Id:           genericParam.Id,
		Name:         genericParam.Name,
		Type:         paramType,
		Options:      options,
		DefaultValue: genericParam.DefaultValue,
	}
}

// MapParameterTypeToGeneric converts a string parameter type to GenericFilterParameterType enum
func (c *FilterCodec) MapParameterTypeToGeneric(paramType string) pb.GenericFilterParameterType {
	switch strings.ToLower(paramType) {
	case paramTypeSelect:
		return pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_SELECT
	case "range", "slider":
		return pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_RANGE
	case "number":
		return pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_NUMBER
	case paramTypeCheckbox:
		return pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_CHECKBOX
	case paramTypeText, "string", "input":
		return pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_TEXT
	default:
		return pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_UNSPECIFIED
	}
}

// MapGenericParameterTypeToString converts GenericFilterParameterType enum to string
func (c *FilterCodec) MapGenericParameterTypeToString(paramType pb.GenericFilterParameterType) string {
	switch paramType {
	case pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_SELECT:
		return paramTypeSelect
	case pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_RANGE:
		return "range"
	case pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_NUMBER:
		return "number"
	case pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_CHECKBOX:
		return paramTypeCheckbox
	case pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_TEXT:
		return paramTypeText
	case pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_UNSPECIFIED:
	default:
		return paramTypeText
	}
	return paramTypeText
}

// BuildParameterMetadata builds metadata map for filter parameters
func (c *FilterCodec) BuildParameterMetadata(filterID string, param *pb.FilterParameter) map[string]string {
	if param == nil {
		return nil
	}

	metadata := make(map[string]string)

	// Temporary heuristics for known parameters until metadata is provided by the accelerator.
	switch {
	case filterID == filterIDBlur && param.Id == "kernel_size":
		metadata["min"] = "3"
		metadata["max"] = "15"
		metadata["step"] = "2"
	case filterID == filterIDBlur && param.Id == "sigma":
		metadata["min"] = "0"
		metadata["max"] = "5"
		metadata["step"] = "0.1"
	case filterID == filterIDBlur && param.Id == "border_mode":
		metadata["display"] = "select"
	case filterID == filterIDBlur && param.Id == "separable":
		metadata["display"] = "checkbox"
	}

	if len(metadata) == 0 {
		return nil
	}
	return metadata
}

// FormatParameterLabel formats parameter option labels for display
func (c *FilterCodec) FormatParameterLabel(value string) string {
	switch strings.ToLower(value) {
	case "bt601":
		return "ITU-R BT.601 (SDTV)"
	case "bt709":
		return "ITU-R BT.709 (HDTV)"
	case "average":
		return "Average"
	case "lightness":
		return "Lightness"
	case "luminosity":
		return "Luminosity"
	default:
		return value
	}
}
