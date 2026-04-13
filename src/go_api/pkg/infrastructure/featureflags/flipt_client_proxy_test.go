package featureflags

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	flipt "go.flipt.io/flipt-client"
)

// Mock FliptClientCore for testing
type mockFliptClientCore struct {
	mock.Mock
}

func (m *mockFliptClientCore) EvaluateBoolean(ctx context.Context, req *flipt.EvaluationRequest) (*flipt.BooleanEvaluationResponse, error) {
	args := m.Called(ctx, req)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*flipt.BooleanEvaluationResponse), args.Error(1)
}

func (m *mockFliptClientCore) EvaluateVariant(ctx context.Context, req *flipt.EvaluationRequest) (*flipt.VariantEvaluationResponse, error) {
	args := m.Called(ctx, req)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*flipt.VariantEvaluationResponse), args.Error(1)
}

func (m *mockFliptClientCore) Close(ctx context.Context) error {
	args := m.Called(ctx)
	return args.Error(0)
}

// Test data builders
func makeValidBooleanResponse(enabled bool) *flipt.BooleanEvaluationResponse {
	return &flipt.BooleanEvaluationResponse{Enabled: enabled}
}

func makeValidVariantResponse(variantKey string) *flipt.VariantEvaluationResponse {
	return &flipt.VariantEvaluationResponse{VariantKey: variantKey}
}

func makeValidEvaluationRequest(flagKey, entityID string) *flipt.EvaluationRequest {
	return &flipt.EvaluationRequest{
		FlagKey:  flagKey,
		EntityID: entityID,
	}
}

// Tests
func TestNewFliptClient(t *testing.T) {
	// Arrange
	mockClient := new(mockFliptClientCore)

	// Act
	sut := NewFliptClient(mockClient)

	// Assert
	require.NotNil(t, sut)
	assert.Equal(t, mockClient, sut.client)
}

func TestFliptClientProxy_EvaluateBoolean(t *testing.T) {
	var errEvaluationError = errors.New("evaluation error")

	tests := []struct {
		name         string
		flagKey      string
		entityID     string
		mockResponse *flipt.BooleanEvaluationResponse
		mockError    error
		assertResult func(t *testing.T, result *flipt.BooleanEvaluationResponse, err error)
	}{
		{
			name:         "Success_BooleanEvaluation",
			flagKey:      "test_flag",
			entityID:     "user123",
			mockResponse: makeValidBooleanResponse(true),
			mockError:    nil,
			assertResult: func(t *testing.T, result *flipt.BooleanEvaluationResponse, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)
				assert.True(t, result.Enabled)
			},
		},
		{
			name:         "Success_DisabledFlag",
			flagKey:      "disabled_flag",
			entityID:     "user456",
			mockResponse: makeValidBooleanResponse(false),
			mockError:    nil,
			assertResult: func(t *testing.T, result *flipt.BooleanEvaluationResponse, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)
				assert.False(t, result.Enabled)
			},
		},
		{
			name:         "Error_EvaluationFails",
			flagKey:      "error_flag",
			entityID:     "user789",
			mockResponse: nil,
			mockError:    errEvaluationError,
			assertResult: func(t *testing.T, result *flipt.BooleanEvaluationResponse, err error) {
				assert.ErrorIs(t, err, errEvaluationError)
				assert.Nil(t, result)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			mockClient := new(mockFliptClientCore)
			mockClient.On("EvaluateBoolean",
				mock.Anything,
				mock.MatchedBy(func(req *flipt.EvaluationRequest) bool {
					return req.FlagKey == tt.flagKey && req.EntityID == tt.entityID
				}),
			).Return(tt.mockResponse, tt.mockError).Once()

			sut := NewFliptClient(mockClient)
			ctx := context.Background()
			req := makeValidEvaluationRequest(tt.flagKey, tt.entityID)

			// Act
			result, err := sut.EvaluateBoolean(ctx, req)

			// Assert
			tt.assertResult(t, result, err)
			mockClient.AssertExpectations(t)
		})
	}
}

func TestFliptClientProxy_EvaluateString(t *testing.T) {
	var errEvaluationError = errors.New("evaluation error")

	tests := []struct {
		name         string
		flagKey      string
		entityID     string
		mockResponse *flipt.VariantEvaluationResponse
		mockError    error
		assertResult func(t *testing.T, result *flipt.VariantEvaluationResponse, err error)
	}{
		{
			name:         "Success_VariantEvaluation",
			flagKey:      "variant_flag",
			entityID:     "user123",
			mockResponse: makeValidVariantResponse("variant_a"),
			mockError:    nil,
			assertResult: func(t *testing.T, result *flipt.VariantEvaluationResponse, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)
				assert.Equal(t, "variant_a", result.VariantKey)
			},
		},
		{
			name:         "Success_DefaultVariant",
			flagKey:      "default_flag",
			entityID:     "user456",
			mockResponse: makeValidVariantResponse("default"),
			mockError:    nil,
			assertResult: func(t *testing.T, result *flipt.VariantEvaluationResponse, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)
				assert.Equal(t, "default", result.VariantKey)
			},
		},
		{
			name:         "Error_EvaluationFails",
			flagKey:      "error_flag",
			entityID:     "user789",
			mockResponse: nil,
			mockError:    errEvaluationError,
			assertResult: func(t *testing.T, result *flipt.VariantEvaluationResponse, err error) {
				assert.ErrorIs(t, err, errEvaluationError)
				assert.Nil(t, result)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			mockClient := new(mockFliptClientCore)
			mockClient.On("EvaluateVariant",
				mock.Anything,
				mock.MatchedBy(func(req *flipt.EvaluationRequest) bool {
					return req.FlagKey == tt.flagKey && req.EntityID == tt.entityID
				}),
			).Return(tt.mockResponse, tt.mockError).Once()

			sut := NewFliptClient(mockClient)
			ctx := context.Background()
			req := makeValidEvaluationRequest(tt.flagKey, tt.entityID)

			// Act
			result, err := sut.EvaluateString(ctx, req)

			// Assert
			tt.assertResult(t, result, err)
			mockClient.AssertExpectations(t)
		})
	}
}

func TestFliptClientProxy_Close(t *testing.T) {
	var errCloseError = errors.New("close error")

	tests := []struct {
		name         string
		mockError    error
		assertResult func(t *testing.T, err error)
	}{
		{
			name:      "Success_ClientClosed",
			mockError: nil,
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name:      "Error_CloseFails",
			mockError: errCloseError,
			assertResult: func(t *testing.T, err error) {
				assert.ErrorIs(t, err, errCloseError)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			mockClient := new(mockFliptClientCore)
			mockClient.On("Close", mock.Anything).Return(tt.mockError).Once()

			sut := NewFliptClient(mockClient)
			ctx := context.Background()

			// Act
			err := sut.Close(ctx)

			// Assert
			tt.assertResult(t, err)
			mockClient.AssertExpectations(t)
		})
	}
}

func TestFliptClientProxy_ContextCancellation(t *testing.T) {
	// Arrange
	mockClient := new(mockFliptClientCore)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	mockClient.On("EvaluateBoolean",
		mock.Anything,
		mock.Anything,
	).Return(nil, ctx.Err()).Once()

	sut := NewFliptClient(mockClient)
	req := makeValidEvaluationRequest("test_flag", "user123")

	// Act
	result, err := sut.EvaluateBoolean(ctx, req)

	// Assert
	assert.Error(t, err)
	assert.Nil(t, result)
	mockClient.AssertExpectations(t)
}

// Test to verify interface compliance
func TestFliptClientProxy_InterfaceCompliance(t *testing.T) {
	// This test ensures that FliptClientProxy implements FliptClientInterface
	var _ FliptClientInterface = (*FliptClientProxy)(nil)
}

// Test to verify that the proxy correctly handles the EvaluateString method
// which internally calls EvaluateVariant on the underlying client
func TestFliptClientProxy_EvaluateStringCallsEvaluateVariant(t *testing.T) {
	// Arrange
	mockClient := new(mockFliptClientCore)
	mockClient.On("EvaluateVariant",
		mock.Anything,
		mock.Anything,
	).Return(makeValidVariantResponse("variant_a"), nil).Once()

	sut := NewFliptClient(mockClient)
	ctx := context.Background()
	req := makeValidEvaluationRequest("variant_flag", "user123")

	// Act
	result, err := sut.EvaluateString(ctx, req)

	// Assert
	assert.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, "variant_a", result.VariantKey)
	mockClient.AssertExpectations(t)
}
