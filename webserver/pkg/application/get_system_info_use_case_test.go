package application

import (
	"context"
	"testing"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// Mock implementations for testing
type MockProcessorRepository struct {
	mock.Mock
}

func (m *MockProcessorRepository) GetAvailableLibraries() []string {
	args := m.Called()
	return args.Get(0).([]string)
}

func (m *MockProcessorRepository) GetCurrentLibrary() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockProcessorRepository) GetAPIVersion() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockProcessorRepository) GetLibraryVersion() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockProcessorRepository) GetLibraryMetadata(version string) (interface{}, error) {
	args := m.Called(version)
	return args.Get(0), args.Error(1)
}

type MockConfigRepository struct {
	mock.Mock
}

func (m *MockConfigRepository) GetEnvironment() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockConfigRepository) GetDefaultLibrary() string {
	args := m.Called()
	return args.String(0)
}

type MockBuildInfoRepository struct {
	mock.Mock
}

func (m *MockBuildInfoRepository) GetVersion() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockBuildInfoRepository) GetBranch() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockBuildInfoRepository) GetBuildTime() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockBuildInfoRepository) GetCommitHash() string {
	args := m.Called()
	return args.String(0)
}

func TestNewGetSystemInfoUseCase(t *testing.T) {
	// Arrange
	mockProcessor := &MockProcessorRepository{}
	mockConfig := &MockConfigRepository{}
	mockBuildInfo := &MockBuildInfoRepository{}

	// Act
	sut := NewGetSystemInfoUseCase(mockProcessor, mockConfig, mockBuildInfo)

	// Assert
	require.NotNil(t, sut)
	assert.Equal(t, mockProcessor, sut.processorRepo)
	assert.Equal(t, mockConfig, sut.configRepo)
	assert.Equal(t, mockBuildInfo, sut.buildInfoRepo)
}

func TestGetSystemInfoUseCase_Execute(t *testing.T) {
	tests := []struct {
		name         string
		setupMocks   func(*MockProcessorRepository, *MockConfigRepository, *MockBuildInfoRepository)
		assertResult func(t *testing.T, result *domain.SystemInfo, err error)
	}{
		{
			name: "Success_AllFieldsPopulated",
			setupMocks: func(processor *MockProcessorRepository, config *MockConfigRepository, buildInfo *MockBuildInfoRepository) {
				processor.On("GetAvailableLibraries").Return([]string{"2.0.0", "1.5.0"})
				processor.On("GetCurrentLibrary").Return("2.0.0")
				processor.On("GetAPIVersion").Return("2.0.0")
				processor.On("GetLibraryVersion").Return("2.0.0")
				config.On("GetEnvironment").Return("development")
				buildInfo.On("GetVersion").Return("1.0.0")
				buildInfo.On("GetBranch").Return("main")
				buildInfo.On("GetBuildTime").Return("2024-10-25T12:00:00Z")
				buildInfo.On("GetCommitHash").Return("abc123")
			},
			assertResult: func(t *testing.T, result *domain.SystemInfo, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)

				// Check version fields
				assert.Equal(t, "1.0.0", result.Version.JsVersion)
				assert.Equal(t, "main", result.Version.Branch)
				assert.Equal(t, "2024-10-25T12:00:00Z", result.Version.BuildTime)
				assert.Equal(t, "abc123", result.Version.CommitHash)

				// Check environment
				assert.Equal(t, "development", result.Environment)

				// Check processor info
				assert.Equal(t, "2.0.0", result.CurrentLibrary)
				assert.Equal(t, "2.0.0", result.APIVersion)
				assert.Equal(t, []string{"2.0.0", "1.5.0"}, result.AvailableLibraries)
			},
		},
		{
			name: "Success_EmptyProcessorInfo",
			setupMocks: func(processor *MockProcessorRepository, config *MockConfigRepository, buildInfo *MockBuildInfoRepository) {
				processor.On("GetAvailableLibraries").Return([]string{})
				processor.On("GetCurrentLibrary").Return("unknown")
				processor.On("GetAPIVersion").Return("unknown")
				processor.On("GetLibraryVersion").Return("unknown")
				config.On("GetEnvironment").Return("production")
				buildInfo.On("GetVersion").Return("1.0.0")
				buildInfo.On("GetBranch").Return("main")
				buildInfo.On("GetBuildTime").Return("2024-10-25T12:00:00Z")
				buildInfo.On("GetCommitHash").Return("abc123")
			},
			assertResult: func(t *testing.T, result *domain.SystemInfo, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)

				assert.Equal(t, "production", result.Environment)
				assert.Equal(t, "unknown", result.CurrentLibrary)
				assert.Equal(t, "unknown", result.APIVersion)
				assert.Empty(t, result.AvailableLibraries)
			},
		},
		{
			name: "Success_CustomValues",
			setupMocks: func(processor *MockProcessorRepository, config *MockConfigRepository, buildInfo *MockBuildInfoRepository) {
				processor.On("GetAvailableLibraries").Return([]string{"3.0.0"})
				processor.On("GetCurrentLibrary").Return("3.0.0")
				processor.On("GetAPIVersion").Return("3.0.0")
				processor.On("GetLibraryVersion").Return("3.0.0")
				config.On("GetEnvironment").Return("staging")
				buildInfo.On("GetVersion").Return("2.5.1")
				buildInfo.On("GetBranch").Return("develop")
				buildInfo.On("GetBuildTime").Return("2024-11-01T10:30:00Z")
				buildInfo.On("GetCommitHash").Return("xyz789")
			},
			assertResult: func(t *testing.T, result *domain.SystemInfo, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)

				assert.Equal(t, "2.5.1", result.Version.JsVersion)
				assert.Equal(t, "develop", result.Version.Branch)
				assert.Equal(t, "2024-11-01T10:30:00Z", result.Version.BuildTime)
				assert.Equal(t, "xyz789", result.Version.CommitHash)
				assert.Equal(t, "staging", result.Environment)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			mockProcessor := &MockProcessorRepository{}
			mockConfig := &MockConfigRepository{}
			mockBuildInfo := &MockBuildInfoRepository{}

			tt.setupMocks(mockProcessor, mockConfig, mockBuildInfo)

			sut := NewGetSystemInfoUseCase(mockProcessor, mockConfig, mockBuildInfo)
			ctx := context.Background()

			// Act
			result, err := sut.Execute(ctx)

			// Assert
			tt.assertResult(t, result, err)
			mockProcessor.AssertExpectations(t)
			mockConfig.AssertExpectations(t)
			mockBuildInfo.AssertExpectations(t)
		})
	}
}
