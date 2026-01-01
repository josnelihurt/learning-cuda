package adapters

import (
	"testing"

	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProtobufAdapter_ToBlurParameters(t *testing.T) {
	adapter := NewProtobufAdapter()

	tests := []struct {
		name     string
		pbBlur   *pb.GaussianBlurParameters
		expected *domain.BlurParameters
	}{
		{
			name:     "Success_WithAllParameters",
			pbBlur:   &pb.GaussianBlurParameters{KernelSize: 5, Sigma: 1.5, BorderMode: pb.BorderMode_BORDER_MODE_REFLECT, Separable: true},
			expected: &domain.BlurParameters{KernelSize: 5, Sigma: 1.5, BorderMode: domain.BorderModeReflect, Separable: true},
		},
		{
			name:     "Success_WithClampBorderMode",
			pbBlur:   &pb.GaussianBlurParameters{KernelSize: 7, Sigma: 2.0, BorderMode: pb.BorderMode_BORDER_MODE_CLAMP, Separable: false},
			expected: &domain.BlurParameters{KernelSize: 7, Sigma: 2.0, BorderMode: domain.BorderModeClamp, Separable: false},
		},
		{
			name:     "Success_WithWrapBorderMode",
			pbBlur:   &pb.GaussianBlurParameters{KernelSize: 9, Sigma: 3.0, BorderMode: pb.BorderMode_BORDER_MODE_WRAP, Separable: true},
			expected: &domain.BlurParameters{KernelSize: 9, Sigma: 3.0, BorderMode: domain.BorderModeWrap, Separable: true},
		},
		{
			name:     "Success_WithUnspecifiedBorderMode_DefaultsToReflect",
			pbBlur:   &pb.GaussianBlurParameters{KernelSize: 5, Sigma: 1.0, BorderMode: pb.BorderMode_BORDER_MODE_UNSPECIFIED, Separable: true},
			expected: &domain.BlurParameters{KernelSize: 5, Sigma: 1.0, BorderMode: domain.BorderModeReflect, Separable: true},
		},
		{
			name:     "Success_NilBlurParams_ReturnsNil",
			pbBlur:   nil,
			expected: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adapter.ToBlurParameters(tt.pbBlur)

			if tt.expected == nil {
				assert.Nil(t, result)
				return
			}

			require.NotNil(t, result)
			assert.Equal(t, tt.expected.KernelSize, result.KernelSize)
			assert.Equal(t, tt.expected.Sigma, result.Sigma)
			assert.Equal(t, tt.expected.BorderMode, result.BorderMode)
			assert.Equal(t, tt.expected.Separable, result.Separable)
		})
	}
}

func TestProtobufAdapter_ToBorderMode(t *testing.T) {
	adapter := NewProtobufAdapter()

	tests := []struct {
		name     string
		pbMode   pb.BorderMode
		expected domain.BorderMode
	}{
		{
			name:     "Success_Clamp",
			pbMode:   pb.BorderMode_BORDER_MODE_CLAMP,
			expected: domain.BorderModeClamp,
		},
		{
			name:     "Success_Reflect",
			pbMode:   pb.BorderMode_BORDER_MODE_REFLECT,
			expected: domain.BorderModeReflect,
		},
		{
			name:     "Success_Wrap",
			pbMode:   pb.BorderMode_BORDER_MODE_WRAP,
			expected: domain.BorderModeWrap,
		},
		{
			name:     "Success_Unspecified_DefaultsToReflect",
			pbMode:   pb.BorderMode_BORDER_MODE_UNSPECIFIED,
			expected: domain.BorderModeReflect,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adapter.ToBorderMode(tt.pbMode)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestProtobufAdapter_ToProtobufBlurParameters(t *testing.T) {
	adapter := NewProtobufAdapter()

	tests := []struct {
		name     string
		blur     *domain.BlurParameters
		expected *pb.GaussianBlurParameters
	}{
		{
			name:     "Success_WithAllParameters",
			blur:     &domain.BlurParameters{KernelSize: 5, Sigma: 1.5, BorderMode: domain.BorderModeReflect, Separable: true},
			expected: &pb.GaussianBlurParameters{KernelSize: 5, Sigma: 1.5, BorderMode: pb.BorderMode_BORDER_MODE_REFLECT, Separable: true},
		},
		{
			name:     "Success_WithClampBorderMode",
			blur:     &domain.BlurParameters{KernelSize: 7, Sigma: 2.0, BorderMode: domain.BorderModeClamp, Separable: false},
			expected: &pb.GaussianBlurParameters{KernelSize: 7, Sigma: 2.0, BorderMode: pb.BorderMode_BORDER_MODE_CLAMP, Separable: false},
		},
		{
			name:     "Success_WithWrapBorderMode",
			blur:     &domain.BlurParameters{KernelSize: 9, Sigma: 3.0, BorderMode: domain.BorderModeWrap, Separable: true},
			expected: &pb.GaussianBlurParameters{KernelSize: 9, Sigma: 3.0, BorderMode: pb.BorderMode_BORDER_MODE_WRAP, Separable: true},
		},
		{
			name:     "Success_NilBlurParams_ReturnsNil",
			blur:     nil,
			expected: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adapter.ToProtobufBlurParameters(tt.blur)

			if tt.expected == nil {
				assert.Nil(t, result)
				return
			}

			require.NotNil(t, result)
			assert.Equal(t, tt.expected.KernelSize, result.KernelSize)
			assert.Equal(t, tt.expected.Sigma, result.Sigma)
			assert.Equal(t, tt.expected.BorderMode, result.BorderMode)
			assert.Equal(t, tt.expected.Separable, result.Separable)
		})
	}
}

func TestProtobufAdapter_ToProtobufBorderMode(t *testing.T) {
	adapter := NewProtobufAdapter()

	tests := []struct {
		name     string
		mode     domain.BorderMode
		expected pb.BorderMode
	}{
		{
			name:     "Success_Clamp",
			mode:     domain.BorderModeClamp,
			expected: pb.BorderMode_BORDER_MODE_CLAMP,
		},
		{
			name:     "Success_Reflect",
			mode:     domain.BorderModeReflect,
			expected: pb.BorderMode_BORDER_MODE_REFLECT,
		},
		{
			name:     "Success_Wrap",
			mode:     domain.BorderModeWrap,
			expected: pb.BorderMode_BORDER_MODE_WRAP,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adapter.ToProtobufBorderMode(tt.mode)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestProtobufAdapter_BlurParametersRoundTrip(t *testing.T) {
	adapter := NewProtobufAdapter()

	tests := []struct {
		name string
		blur *domain.BlurParameters
	}{
		{
			name: "Success_RoundTrip_AllModes",
			blur: &domain.BlurParameters{KernelSize: 5, Sigma: 1.0, BorderMode: domain.BorderModeReflect, Separable: true},
		},
		{
			name: "Success_RoundTrip_Clamp",
			blur: &domain.BlurParameters{KernelSize: 7, Sigma: 2.0, BorderMode: domain.BorderModeClamp, Separable: false},
		},
		{
			name: "Success_RoundTrip_Wrap",
			blur: &domain.BlurParameters{KernelSize: 9, Sigma: 3.0, BorderMode: domain.BorderModeWrap, Separable: true},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pbBlur := adapter.ToProtobufBlurParameters(tt.blur)
			domainBlur := adapter.ToBlurParameters(pbBlur)

			require.NotNil(t, domainBlur)
			assert.Equal(t, tt.blur.KernelSize, domainBlur.KernelSize)
			assert.Equal(t, tt.blur.Sigma, domainBlur.Sigma)
			assert.Equal(t, tt.blur.BorderMode, domainBlur.BorderMode)
			assert.Equal(t, tt.blur.Separable, domainBlur.Separable)
		})
	}
}

func TestProtobufAdapter_ExtractProcessingOptions_GenericSelections(t *testing.T) {
	t.Parallel()

	adapter := NewProtobufAdapter()
	req := &pb.ProcessImageRequest{
		GenericFilters: []*pb.GenericFilterSelection{
			{
				FilterId: "grayscale",
				Parameters: []*pb.GenericFilterParameterSelection{
					{
						ParameterId: "algorithm",
						Values:      []string{"bt709"},
					},
				},
			},
			{
				FilterId: "blur",
				Parameters: []*pb.GenericFilterParameterSelection{
					{
						ParameterId: "kernel_size",
						Values:      []string{"9"},
					},
					{
						ParameterId: "sigma",
						Values:      []string{"2.5"},
					},
					{
						ParameterId: "border_mode",
						Values:      []string{"wrap"},
					},
					{
						ParameterId: "separable",
						Values:      []string{"false"},
					},
				},
			},
		},
	}

	opts := adapter.ExtractProcessingOptions(req)

	require.Len(t, opts.Filters, 2)
	assert.Equal(t, domain.FilterGrayscale, opts.Filters[0])
	assert.Equal(t, domain.FilterBlur, opts.Filters[1])
	assert.Equal(t, domain.GrayscaleBT709, opts.GrayscaleType)

	require.NotNil(t, opts.BlurParams)
	assert.Equal(t, int32(9), opts.BlurParams.KernelSize)
	assert.InDelta(t, 2.5, opts.BlurParams.Sigma, 0.001)
	assert.Equal(t, domain.BorderModeWrap, opts.BlurParams.BorderMode)
	assert.False(t, opts.BlurParams.Separable)
}

func TestProtobufAdapter_ExtractProcessingOptions_FallbackToLegacyFields(t *testing.T) {
	t.Parallel()

	adapter := NewProtobufAdapter()
	req := &pb.ProcessImageRequest{
		Filters: []pb.FilterType{
			pb.FilterType_FILTER_TYPE_BLUR,
		},
		GrayscaleType: pb.GrayscaleType_GRAYSCALE_TYPE_LIGHTNESS,
		BlurParams: &pb.GaussianBlurParameters{
			KernelSize: 11,
			Sigma:      0.9,
			BorderMode: pb.BorderMode_BORDER_MODE_CLAMP,
			Separable:  false,
		},
		GenericFilters: []*pb.GenericFilterSelection{
			{
				FilterId: "unknown-filter",
			},
		},
	}

	opts := adapter.ExtractProcessingOptions(req)

	require.Len(t, opts.Filters, 1)
	assert.Equal(t, domain.FilterBlur, opts.Filters[0])
	assert.Equal(t, domain.GrayscaleLightness, opts.GrayscaleType)
	require.NotNil(t, opts.BlurParams)
	assert.Equal(t, int32(11), opts.BlurParams.KernelSize)
	assert.Equal(t, float32(0.9), opts.BlurParams.Sigma)
	assert.Equal(t, domain.BorderModeClamp, opts.BlurParams.BorderMode)
	assert.False(t, opts.BlurParams.Separable)
}
