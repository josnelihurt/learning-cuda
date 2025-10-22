package featureflags

import (
	"context"
	"errors"
	"testing"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	flipt "go.flipt.io/flipt-client"
)

// Interface for FliptWriter to enable mocking
type FliptWriterInterface interface {
	SyncFlags(ctx context.Context, flags map[string]interface{}) error
}

// Mocks
type mockFliptClient struct {
	mock.Mock
}

func (m *mockFliptClient) EvaluateBoolean(ctx context.Context, req *flipt.EvaluationRequest) (*flipt.BooleanEvaluationResponse, error) {
	args := m.Called(ctx, req)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*flipt.BooleanEvaluationResponse), args.Error(1)
}

func (m *mockFliptClient) EvaluateString(ctx context.Context, req *flipt.EvaluationRequest) (*flipt.VariantEvaluationResponse, error) {
	args := m.Called(ctx, req)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*flipt.VariantEvaluationResponse), args.Error(1)
}

func (m *mockFliptClient) Close(ctx context.Context) error {
	args := m.Called(ctx)
	return args.Error(0)
}

// Builders
func makeValidBooleanResponse(enabled bool) *flipt.BooleanEvaluationResponse {
	return &flipt.BooleanEvaluationResponse{Enabled: enabled}
}

func makeValidVariantResponse(variant string) *flipt.VariantEvaluationResponse {
	return &flipt.VariantEvaluationResponse{VariantAttachment: variant}
}

func makeValidFeatureFlag(key string, defaultValue interface{}) domain.FeatureFlag {
	return domain.FeatureFlag{
		Key:          key,
		Name:         "Test Flag",
		Type:         domain.BooleanFlagType,
		Enabled:      true,
		DefaultValue: defaultValue,
	}
}

// Tests
func TestNewFliptRepository(t *testing.T) {
	// Arrange
	mockClient := new(mockFliptClient)
	writer := &FliptWriter{}

	// Act
	sut := NewFliptRepository(mockClient, writer)

	// Assert
	require.NotNil(t, sut)
	assert.Equal(t, mockClient, sut.reader)
	assert.Equal(t, writer, sut.writer)
}

func TestFliptRepository_EvaluateBoolean(t *testing.T) {
	var errFliptError = errors.New("flipt error")

	tests := []struct {
		name         string
		flagKey      string
		entityID     string
		reader       FliptClientInterface
		mockResponse *flipt.BooleanEvaluationResponse
		mockError    error
		assertResult func(t *testing.T, result *domain.FeatureFlagEvaluation, err error)
	}{
		{
			name:         "Success_FlagEnabled",
			flagKey:      "test_flag",
			entityID:     "user123",
			reader:       new(mockFliptClient),
			mockResponse: makeValidBooleanResponse(true),
			mockError:    nil,
			assertResult: func(t *testing.T, result *domain.FeatureFlagEvaluation, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)
				assert.Equal(t, "test_flag", result.FlagKey)
				assert.Equal(t, "user123", result.EntityID)
				assert.Equal(t, true, result.Result)
				assert.True(t, result.Success)
				assert.False(t, result.UsedFallback)
			},
		},
		{
			name:         "Success_FlagDisabled",
			flagKey:      "disabled_flag",
			entityID:     "user456",
			reader:       new(mockFliptClient),
			mockResponse: makeValidBooleanResponse(false),
			mockError:    nil,
			assertResult: func(t *testing.T, result *domain.FeatureFlagEvaluation, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)
				assert.Equal(t, "disabled_flag", result.FlagKey)
				assert.Equal(t, "user456", result.EntityID)
				assert.Equal(t, false, result.Result)
				assert.True(t, result.Success)
				assert.False(t, result.UsedFallback)
			},
		},
		{
			name:         "Success_NilReader_ReturnsFallback",
			flagKey:      "test_flag",
			entityID:     "user123",
			reader:       nil,
			mockResponse: nil,
			mockError:    nil,
			assertResult: func(t *testing.T, result *domain.FeatureFlagEvaluation, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)
				assert.Equal(t, "test_flag", result.FlagKey)
				assert.Equal(t, "user123", result.EntityID)
				assert.False(t, result.Success)
				assert.True(t, result.UsedFallback)
			},
		},
		{
			name:         "Error_FliptClientError_ReturnsFallback",
			flagKey:      "error_flag",
			entityID:     "user789",
			reader:       new(mockFliptClient),
			mockResponse: nil,
			mockError:    errFliptError,
			assertResult: func(t *testing.T, result *domain.FeatureFlagEvaluation, err error) {
				assert.ErrorIs(t, err, errFliptError)
				require.NotNil(t, result)
				assert.Equal(t, "error_flag", result.FlagKey)
				assert.Equal(t, "user789", result.EntityID)
				assert.False(t, result.Success)
				assert.True(t, result.UsedFallback)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			var reader FliptClientInterface
			if tt.reader != nil {
				mockClient := new(mockFliptClient)
				mockClient.On("EvaluateBoolean",
					mock.Anything,
					mock.MatchedBy(func(req *flipt.EvaluationRequest) bool {
						return req.FlagKey == tt.flagKey && req.EntityID == tt.entityID
					}),
				).Return(tt.mockResponse, tt.mockError).Once()
				reader = mockClient
			}

			sut := NewFliptRepository(reader, nil)
			ctx := context.Background()

			// Act
			result, err := sut.EvaluateBoolean(ctx, tt.flagKey, tt.entityID)

			// Assert
			tt.assertResult(t, result, err)
			if reader != nil {
				reader.(*mockFliptClient).AssertExpectations(t)
			}
		})
	}
}

func TestFliptRepository_EvaluateVariant(t *testing.T) {
	var errFliptError = errors.New("flipt error")

	tests := []struct {
		name         string
		flagKey      string
		entityID     string
		reader       FliptClientInterface
		mockResponse *flipt.VariantEvaluationResponse
		mockError    error
		assertResult func(t *testing.T, result *domain.FeatureFlagEvaluation, err error)
	}{
		{
			name:         "Success_ReturnsVariant",
			flagKey:      "variant_flag",
			entityID:     "user123",
			reader:       new(mockFliptClient),
			mockResponse: makeValidVariantResponse("variant_a"),
			mockError:    nil,
			assertResult: func(t *testing.T, result *domain.FeatureFlagEvaluation, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)
				assert.Equal(t, "variant_flag", result.FlagKey)
				assert.Equal(t, "user123", result.EntityID)
				assert.Equal(t, "variant_a", result.Result)
				assert.True(t, result.Success)
				assert.False(t, result.UsedFallback)
			},
		},
		{
			name:         "Success_NilReader_ReturnsFallback",
			flagKey:      "variant_flag",
			entityID:     "user123",
			reader:       nil,
			mockResponse: nil,
			mockError:    nil,
			assertResult: func(t *testing.T, result *domain.FeatureFlagEvaluation, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)
				assert.Equal(t, "variant_flag", result.FlagKey)
				assert.Equal(t, "user123", result.EntityID)
				assert.False(t, result.Success)
				assert.True(t, result.UsedFallback)
			},
		},
		{
			name:         "Error_FliptClientError_ReturnsFallback",
			flagKey:      "error_variant",
			entityID:     "user456",
			reader:       new(mockFliptClient),
			mockResponse: nil,
			mockError:    errFliptError,
			assertResult: func(t *testing.T, result *domain.FeatureFlagEvaluation, err error) {
				assert.ErrorIs(t, err, errFliptError)
				require.NotNil(t, result)
				assert.Equal(t, "error_variant", result.FlagKey)
				assert.Equal(t, "user456", result.EntityID)
				assert.False(t, result.Success)
				assert.True(t, result.UsedFallback)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			var reader FliptClientInterface
			if tt.reader != nil {
				mockClient := new(mockFliptClient)
				mockClient.On("EvaluateString",
					mock.Anything,
					mock.MatchedBy(func(req *flipt.EvaluationRequest) bool {
						return req.FlagKey == tt.flagKey && req.EntityID == tt.entityID
					}),
				).Return(tt.mockResponse, tt.mockError).Once()
				reader = mockClient
			}

			sut := NewFliptRepository(reader, nil)
			ctx := context.Background()

			// Act
			result, err := sut.EvaluateVariant(ctx, tt.flagKey, tt.entityID)

			// Assert
			tt.assertResult(t, result, err)
			if reader != nil {
				reader.(*mockFliptClient).AssertExpectations(t)
			}
		})
	}
}

func TestFliptRepository_SyncFlags(t *testing.T) {
	tests := []struct {
		name         string
		flags        []domain.FeatureFlag
		writer       *FliptWriter
		assertResult func(t *testing.T, err error)
	}{
		{
			name:   "Success_NilWriter_SkipsSync",
			flags:  []domain.FeatureFlag{makeValidFeatureFlag("flag1", true)},
			writer: nil,
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name:   "Edge_EmptyFlagList",
			flags:  []domain.FeatureFlag{},
			writer: &FliptWriter{},
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			sut := NewFliptRepository(nil, tt.writer)
			ctx := context.Background()

			// Act
			err := sut.SyncFlags(ctx, tt.flags)

			// Assert
			tt.assertResult(t, err)
		})
	}
}

func TestFliptRepository_GetFlag(t *testing.T) {
	// Arrange
	sut := NewFliptRepository(nil, nil)
	ctx := context.Background()

	// Act
	result, err := sut.GetFlag(ctx, "test_flag")

	// Assert
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "GetFlag not implemented")
	assert.Nil(t, result)
}
