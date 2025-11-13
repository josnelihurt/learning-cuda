package application

import (
	"context"
	"testing"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

type MockConfigRepository struct {
	mock.Mock
}

func (m *MockConfigRepository) GetEnvironment() string {
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

type MockVersionRepository struct {
	mock.Mock
}

func (m *MockVersionRepository) GetGoVersion() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockVersionRepository) GetCppVersion() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockVersionRepository) GetProtoVersion() string {
	args := m.Called()
	return args.String(0)
}

func TestNewGetSystemInfoUseCase(t *testing.T) {
	// Arrange
	mockConfig := &MockConfigRepository{}
	mockBuildInfo := &MockBuildInfoRepository{}
	mockVersion := &MockVersionRepository{}

	// Act
	sut := NewGetSystemInfoUseCase(mockConfig, mockBuildInfo, mockVersion)

	// Assert
	require.NotNil(t, sut)
	assert.Equal(t, mockConfig, sut.configRepo)
	assert.Equal(t, mockBuildInfo, sut.buildInfoRepo)
	assert.Equal(t, mockVersion, sut.versionRepo)
}

func TestGetSystemInfoUseCase_Execute(t *testing.T) {
	tests := []struct {
		name         string
		setupMocks   func(*MockConfigRepository, *MockBuildInfoRepository, *MockVersionRepository)
		assertResult func(t *testing.T, result *domain.SystemInfo, err error)
	}{
		{
			name: "Success_AllFieldsPopulated",
			setupMocks: func(config *MockConfigRepository, buildInfo *MockBuildInfoRepository, version *MockVersionRepository) {
				config.On("GetEnvironment").Return("development")
				buildInfo.On("GetBranch").Return("main")
				buildInfo.On("GetBuildTime").Return("2024-10-25T12:00:00Z")
				buildInfo.On("GetCommitHash").Return("abc123")
				version.On("GetGoVersion").Return("1.0.8")
				version.On("GetCppVersion").Return("2.1.6")
				version.On("GetProtoVersion").Return("1.0.0")
			},
			assertResult: func(t *testing.T, result *domain.SystemInfo, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)

				// Check version fields
				assert.Equal(t, "1.0.8", result.Version.GoVersion)
				assert.Equal(t, "2.1.6", result.Version.CppVersion)
				assert.Equal(t, "1.0.0", result.Version.ProtoVersion)
				assert.Equal(t, "main", result.Version.Branch)
				assert.Equal(t, "2024-10-25T12:00:00Z", result.Version.BuildTime)
				assert.Equal(t, "abc123", result.Version.CommitHash)

				// Check environment
				assert.Equal(t, "development", result.Environment)
			},
		},
		{
			name: "Success_ProductionEnvironment",
			setupMocks: func(config *MockConfigRepository, buildInfo *MockBuildInfoRepository, version *MockVersionRepository) {
				config.On("GetEnvironment").Return("production")
				buildInfo.On("GetBranch").Return("main")
				buildInfo.On("GetBuildTime").Return("2024-10-25T12:00:00Z")
				buildInfo.On("GetCommitHash").Return("abc123")
				version.On("GetGoVersion").Return("1.0.8")
				version.On("GetCppVersion").Return("2.1.6")
				version.On("GetProtoVersion").Return("1.0.0")
			},
			assertResult: func(t *testing.T, result *domain.SystemInfo, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)

				assert.Equal(t, "1.0.8", result.Version.GoVersion)
				assert.Equal(t, "2.1.6", result.Version.CppVersion)
				assert.Equal(t, "1.0.0", result.Version.ProtoVersion)
				assert.Equal(t, "production", result.Environment)
			},
		},
		{
			name: "Success_CustomValues",
			setupMocks: func(config *MockConfigRepository, buildInfo *MockBuildInfoRepository, version *MockVersionRepository) {
				config.On("GetEnvironment").Return("staging")
				buildInfo.On("GetBranch").Return("develop")
				buildInfo.On("GetBuildTime").Return("2024-11-01T10:30:00Z")
				buildInfo.On("GetCommitHash").Return("xyz789")
				version.On("GetGoVersion").Return("2.0.0")
				version.On("GetCppVersion").Return("3.0.0")
				version.On("GetProtoVersion").Return("2.0.0")
			},
			assertResult: func(t *testing.T, result *domain.SystemInfo, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)

				assert.Equal(t, "2.0.0", result.Version.GoVersion)
				assert.Equal(t, "3.0.0", result.Version.CppVersion)
				assert.Equal(t, "2.0.0", result.Version.ProtoVersion)
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
			mockConfig := &MockConfigRepository{}
			mockBuildInfo := &MockBuildInfoRepository{}
			mockVersion := &MockVersionRepository{}

			tt.setupMocks(mockConfig, mockBuildInfo, mockVersion)

			sut := NewGetSystemInfoUseCase(mockConfig, mockBuildInfo, mockVersion)
			ctx := context.Background()

			// Act
			result, err := sut.Execute(ctx)

			// Assert
			tt.assertResult(t, result, err)
			mockConfig.AssertExpectations(t)
			mockBuildInfo.AssertExpectations(t)
			mockVersion.AssertExpectations(t)
		})
	}
}
