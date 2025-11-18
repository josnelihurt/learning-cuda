package connectrpc

import (
	"context"
	"errors"
	"testing"

	"connectrpc.com/connect"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockCapabilitiesProvider struct {
	capabilities *pb.LibraryCapabilities
	origin       application.ProcessorBackendOrigin
}

func (m *mockCapabilitiesProvider) Execute(_ context.Context, _ bool) (*pb.LibraryCapabilities, application.ProcessorBackendOrigin, error) {
	if m.capabilities == nil {
		return nil, "", errors.New("capabilities not available")
	}
	if m.origin == "" {
		m.origin = application.ProcessorBackendOriginCGO
	}
	return m.capabilities, m.origin, nil
}

func makeLibraryCapabilities() *pb.LibraryCapabilities {
	return &pb.LibraryCapabilities{
		ApiVersion: "2.1.0",
		Filters: []*pb.FilterDefinition{
			makeGrayscaleFilterDefinition(),
			makeBlurFilterDefinition(),
		},
	}
}

func makeGrayscaleFilterDefinition() *pb.FilterDefinition {
	return &pb.FilterDefinition{
		Id:   "grayscale",
		Name: "Grayscale",
		Parameters: []*pb.FilterParameter{
			{
				Id:           "algorithm",
				Name:         "Algorithm",
				Type:         "select",
				Options:      []string{"bt601", "bt709", "average"},
				DefaultValue: "bt601",
			},
		},
		SupportedAccelerators: []pb.AcceleratorType{
			pb.AcceleratorType_ACCELERATOR_TYPE_CUDA,
			pb.AcceleratorType_ACCELERATOR_TYPE_CPU,
		},
	}
}

func makeBlurFilterDefinition() *pb.FilterDefinition {
	return &pb.FilterDefinition{
		Id:   "blur",
		Name: "Gaussian Blur",
		Parameters: []*pb.FilterParameter{
			{
				Id:           "kernel_size",
				Name:         "Kernel Size",
				Type:         "range",
				DefaultValue: "5",
			},
			{
				Id:           "sigma",
				Name:         "Sigma",
				Type:         "number",
				DefaultValue: "1.0",
			},
		},
		SupportedAccelerators: []pb.AcceleratorType{
			pb.AcceleratorType_ACCELERATOR_TYPE_CUDA,
		},
	}
}

func TestImageProcessorHandler_ListFilters(t *testing.T) {
	tests := []struct {
		name         string
		provider     application.ProcessorCapabilitiesUseCase
		assertResult func(t *testing.T, resp *connect.Response[pb.ListFiltersResponse], err error)
	}{
		{
			name:     "Success_ReturnsGenericFilters",
			provider: &mockCapabilitiesProvider{capabilities: makeLibraryCapabilities(), origin: application.ProcessorBackendOriginCGO},
			assertResult: func(t *testing.T, resp *connect.Response[pb.ListFiltersResponse], err error) {
				// Assert
				require.NoError(t, err)
				require.NotNil(t, resp)
				require.NotNil(t, resp.Msg)

				assert.Equal(t, "2.1.0", resp.Msg.ApiVersion)
				require.Len(t, resp.Msg.Filters, 2)

				grayscale := resp.Msg.Filters[0]
				assert.Equal(t, "grayscale", grayscale.Id)
				require.Len(t, grayscale.Parameters, 1)
				assert.Equal(t, pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_SELECT, grayscale.Parameters[0].Type)
				require.Len(t, grayscale.Parameters[0].Options, 3)

				blur := resp.Msg.Filters[1]
				assert.Equal(t, "blur", blur.Id)
				require.Len(t, blur.Parameters, 2)
				assert.Equal(t, pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_RANGE, blur.Parameters[0].Type)
				assert.Equal(t, pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_NUMBER, blur.Parameters[1].Type)

				assert.Equal(t, "cgo", resp.Header().Get("X-Processor-Backend"))
			},
		},
		{
			name:     "Error_CapabilitiesUnavailable",
			provider: &mockCapabilitiesProvider{capabilities: nil},
			assertResult: func(t *testing.T, resp *connect.Response[pb.ListFiltersResponse], err error) {
				// Assert
				require.Error(t, err)
				assert.Nil(t, resp)

				var connectErr *connect.Error
				require.True(t, errors.As(err, &connectErr))
				assert.Equal(t, connect.CodeUnavailable, connectErr.Code())
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			sut := NewImageProcessorHandler((*application.ProcessImageUseCase)(nil), tt.provider, nil, false)
			req := connect.NewRequest(&pb.ListFiltersRequest{})

			// Act
			resp, err := sut.ListFilters(context.Background(), req)

			// Assert
			tt.assertResult(t, resp, err)
		})
	}
}
