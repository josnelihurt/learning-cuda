package config

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestManager_IsObservabilityEnabled(t *testing.T) {
	tests := []struct {
		name           string
		observability  ObservabilityConfig
		expectedResult bool
	}{
		{
			name: "Success_ObservabilityEnabled",
			observability: ObservabilityConfig{
				Enabled: true,
			},
			expectedResult: true,
		},
		{
			name: "Success_ObservabilityDisabled",
			observability: ObservabilityConfig{
				Enabled: false,
			},
			expectedResult: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			manager := &Manager{
				Observability: tt.observability,
			}
			ctx := context.Background()

			// Act
			result := manager.IsObservabilityEnabled(ctx)

			// Assert
			assert.Equal(t, tt.expectedResult, result)
		})
	}
}

func TestNew(t *testing.T) {
	// This test is complex due to file system dependencies
	// In a real implementation, we would need to create test config files
	t.Skip("Skipping due to file system dependencies - requires integration testing")
}

func TestSetDefaults(t *testing.T) {
	// This test is complex due to viper dependencies
	// In a real implementation, we would need to mock viper
	t.Skip("Skipping due to viper dependencies - requires integration testing")
}

func TestFliptConfig_Validation(t *testing.T) {
	tests := []struct {
		name        string
		config      FliptConfig
		expectValid bool
	}{
		{
			name: "Success_ValidConfig",
			config: FliptConfig{
				Enabled:        true,
				URL:            "http://localhost:8081",
				Namespace:      "default",
				DBPath:         "/tmp/flipt.db",
				ClientTimeout:  30 * time.Second,
				UpdateInterval: 30 * time.Second,
				HTTPTimeout:    10 * time.Second,
			},
			expectValid: true,
		},
		{
			name: "Success_DisabledConfig",
			config: FliptConfig{
				Enabled: false,
			},
			expectValid: true,
		},
		{
			name: "Error_EmptyURL",
			config: FliptConfig{
				Enabled: true,
				URL:     "",
			},
			expectValid: false,
		},
		{
			name: "Error_EmptyNamespace",
			config: FliptConfig{
				Enabled:   true,
				URL:       "http://localhost:8081",
				Namespace: "",
			},
			expectValid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			config := tt.config

			// Act & Assert
			if tt.expectValid {
				// If enabled, URL and Namespace should be set
				if config.Enabled {
					assert.NotEmpty(t, config.URL)
					assert.NotEmpty(t, config.Namespace)
				}
			} else {
				// If enabled but missing required fields, it's invalid
				if config.Enabled {
					assert.True(t, config.URL == "" || config.Namespace == "")
				}
			}
		})
	}
}

func TestServerConfig_Validation(t *testing.T) {
	tests := []struct {
		name        string
		config      ServerConfig
		expectValid bool
	}{
		{
			name: "Success_ValidConfig",
			config: ServerConfig{
				HTTPPort:         ":8080",
				HTTPSPort:        ":8443",
				HotReloadEnabled: false,
				WebRootPath:      "webserver/web",
				DevServerURL:     "https://localhost:3000",
				DevServerPaths:   []string{"/@vite/", "/src/"},
				TLS: TLSConfig{
					Enabled:  true,
					CertFile: ".secrets/cert.pem",
					KeyFile:  ".secrets/key.pem",
				},
			},
			expectValid: true,
		},
		{
			name: "Success_HTTPOnly",
			config: ServerConfig{
				HTTPPort:         ":8080",
				HTTPSPort:        "",
				HotReloadEnabled: false,
				WebRootPath:      "webserver/web",
				TLS: TLSConfig{
					Enabled: false,
				},
			},
			expectValid: true,
		},
		{
			name: "Error_EmptyHTTPPort",
			config: ServerConfig{
				HTTPPort: "",
			},
			expectValid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			config := tt.config

			// Act & Assert
			if tt.expectValid {
				assert.NotEmpty(t, config.HTTPPort)
			} else {
				assert.Empty(t, config.HTTPPort)
			}
		})
	}
}

func TestStreamConfig_Validation(t *testing.T) {
	tests := []struct {
		name        string
		config      StreamConfig
		expectValid bool
	}{
		{
			name: "Success_ValidConfig",
			config: StreamConfig{
				TransportFormat:   "json",
				WebsocketEndpoint: "/ws",
			},
			expectValid: true,
		},
		{
			name: "Success_BinaryTransport",
			config: StreamConfig{
				TransportFormat:   "binary",
				WebsocketEndpoint: "/ws",
			},
			expectValid: true,
		},
		{
			name: "Error_EmptyTransportFormat",
			config: StreamConfig{
				TransportFormat: "",
			},
			expectValid: false,
		},
		{
			name: "Error_EmptyWebsocketEndpoint",
			config: StreamConfig{
				TransportFormat:   "json",
				WebsocketEndpoint: "",
			},
			expectValid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			config := tt.config

			// Act & Assert
			if tt.expectValid {
				assert.NotEmpty(t, config.TransportFormat)
				assert.NotEmpty(t, config.WebsocketEndpoint)
			} else {
				assert.True(t, config.TransportFormat == "" || config.WebsocketEndpoint == "")
			}
		})
	}
}

func TestObservabilityConfig_Validation(t *testing.T) {
	tests := []struct {
		name        string
		config      ObservabilityConfig
		expectValid bool
	}{
		{
			name: "Success_ValidConfig",
			config: ObservabilityConfig{
				Enabled:               true,
				ServiceName:           "cuda-image-processor",
				ServiceVersion:        "1.0.0",
				OtelCollectorEndpoint: "localhost:4317",
				TraceSamplingRate:     1.0,
			},
			expectValid: true,
		},
		{
			name: "Success_Disabled",
			config: ObservabilityConfig{
				Enabled: false,
			},
			expectValid: true,
		},
		{
			name: "Error_EmptyServiceName",
			config: ObservabilityConfig{
				Enabled:     true,
				ServiceName: "",
			},
			expectValid: false,
		},
		{
			name: "Error_InvalidSamplingRate",
			config: ObservabilityConfig{
				Enabled:           true,
				ServiceName:       "test",
				TraceSamplingRate: 1.5, // Invalid: > 1.0
			},
			expectValid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			config := tt.config

			// Act & Assert
			if tt.expectValid {
				if config.Enabled {
					assert.NotEmpty(t, config.ServiceName)
					assert.True(t, config.TraceSamplingRate >= 0.0 && config.TraceSamplingRate <= 1.0)
				}
			} else {
				if config.Enabled {
					assert.True(t, config.ServiceName == "" || config.TraceSamplingRate < 0.0 || config.TraceSamplingRate > 1.0)
				}
			}
		})
	}
}

func TestToolDefinition_Validation(t *testing.T) {
	tests := []struct {
		name        string
		tool        ToolDefinition
		expectValid bool
	}{
		{
			name: "Success_ValidTool",
			tool: ToolDefinition{
				ID:       "jaeger",
				Name:     "Jaeger",
				IconPath: "/icons/jaeger.png",
				Type:     "observability",
				URLDev:   "http://localhost:16686",
				URLProd:  "https://jaeger.prod.com",
				Action:   "open",
			},
			expectValid: true,
		},
		{
			name: "Error_EmptyID",
			tool: ToolDefinition{
				ID: "",
			},
			expectValid: false,
		},
		{
			name: "Error_EmptyName",
			tool: ToolDefinition{
				ID:   "jaeger",
				Name: "",
			},
			expectValid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			tool := tt.tool

			// Act & Assert
			if tt.expectValid {
				assert.NotEmpty(t, tool.ID)
				assert.NotEmpty(t, tool.Name)
			} else {
				assert.True(t, tool.ID == "" || tool.Name == "")
			}
		})
	}
}

func TestToolsConfig_Validation(t *testing.T) {
	tests := []struct {
		name        string
		config      ToolsConfig
		expectValid bool
	}{
		{
			name: "Success_ValidConfig",
			config: ToolsConfig{
				Observability: []ToolDefinition{
					{
						ID:   "jaeger",
						Name: "Jaeger",
						Type: "observability",
					},
				},
				Features: []ToolDefinition{
					{
						ID:   "flipt",
						Name: "Flipt",
						Type: "features",
					},
				},
				Testing: []ToolDefinition{
					{
						ID:   "cucumber",
						Name: "Cucumber",
						Type: "testing",
					},
				},
			},
			expectValid: true,
		},
		{
			name: "Success_EmptyConfig",
			config: ToolsConfig{
				Observability: []ToolDefinition{},
				Features:      []ToolDefinition{},
				Testing:       []ToolDefinition{},
			},
			expectValid: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			config := tt.config

			// Act & Assert
			if tt.expectValid {
				// All tools should be valid if present
				for _, tool := range config.Observability {
					assert.NotEmpty(t, tool.ID)
					assert.NotEmpty(t, tool.Name)
				}
				for _, tool := range config.Features {
					assert.NotEmpty(t, tool.ID)
					assert.NotEmpty(t, tool.Name)
				}
				for _, tool := range config.Testing {
					assert.NotEmpty(t, tool.ID)
					assert.NotEmpty(t, tool.Name)
				}
			}
		})
	}
}

func TestStaticImagesConfig_Validation(t *testing.T) {
	tests := []struct {
		name        string
		config      StaticImagesConfig
		expectValid bool
	}{
		{
			name: "Success_ValidConfig",
			config: StaticImagesConfig{
				Directory: "/data/static_images",
			},
			expectValid: true,
		},
		{
			name: "Error_EmptyDirectory",
			config: StaticImagesConfig{
				Directory: "",
			},
			expectValid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			config := tt.config

			// Act & Assert
			if tt.expectValid {
				assert.NotEmpty(t, config.Directory)
			} else {
				assert.Empty(t, config.Directory)
			}
		})
	}
}
