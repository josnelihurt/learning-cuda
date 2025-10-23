package featureflags

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// Mock HTTP client for testing
type mockHTTPClient struct {
	mock.Mock
}

func (m *mockHTTPClient) Do(req *http.Request) (*http.Response, error) {
	args := m.Called(req)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*http.Response), args.Error(1)
}

// Test data builders
func makeValidFlagsMap() map[string]interface{} {
	return map[string]interface{}{
		"boolean_flag": true,
		"variant_flag": "variant_a",
		"empty_flag":   "",
	}
}

// Tests
func TestNewFliptWriter(t *testing.T) {
	// Arrange
	grpcURL := "grpc://localhost:9000"
	namespace := "default"
	mockClient := new(mockHTTPClient)

	// Act
	sut := NewFliptWriter(grpcURL, namespace, mockClient)

	// Assert
	require.NotNil(t, sut)
	assert.Equal(t, grpcURL, sut.apiURL)
	assert.Equal(t, namespace, sut.namespace)
	assert.Equal(t, mockClient, sut.client)
}

func TestConvertGRPCToRESTURL(t *testing.T) {
	// Arrange
	grpcURL := "grpc://localhost:9000"

	// Act
	result := convertGRPCToRESTURL(grpcURL)

	// Assert
	assert.Equal(t, grpcURL, result)
}

// Helper function to create complex server handlers
func createComplexServerHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/flags/"):
			w.WriteHeader(http.StatusNotFound)
		case r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/flags"):
			w.WriteHeader(http.StatusCreated)
		case r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/variants"):
			w.WriteHeader(http.StatusCreated)
		default:
			w.WriteHeader(http.StatusOK)
		}
	}
}

func TestFliptWriter_SyncFlags(t *testing.T) {
	tests := []struct {
		name         string
		flags        map[string]interface{}
		serverSetup  func() *httptest.Server
		assertResult func(t *testing.T, err error)
	}{
		{
			name:  "Success_BooleanFlagsSync",
			flags: map[string]interface{}{"test_flag": true},
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(createComplexServerHandler())
			},
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name:  "Success_VariantFlagsSync",
			flags: map[string]interface{}{"variant_flag": "variant_a"},
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(createComplexServerHandler())
			},
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name:  "Success_MixedFlagTypes",
			flags: makeValidFlagsMap(),
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					switch {
					case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/flags/"):
						w.WriteHeader(http.StatusNotFound)
					case r.Method == http.MethodPost:
						w.WriteHeader(http.StatusCreated)
					default:
						w.WriteHeader(http.StatusOK)
					}
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name:  "Edge_EmptyFlagsList",
			flags: map[string]interface{}{},
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusOK)
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name:  "Edge_EmptyStringValue",
			flags: map[string]interface{}{"empty_flag": ""},
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusOK)
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name:  "Error_HTTPFailure",
			flags: map[string]interface{}{"test_flag": true},
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusInternalServerError)
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "failed to sync")
			},
		},
		{
			name:  "Error_UnsupportedFlagType",
			flags: map[string]interface{}{"unsupported_flag": 123},
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusOK)
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			server := tt.serverSetup()
			defer server.Close()

			sut := NewFliptWriter(server.URL, "default", &http.Client{})
			ctx := context.Background()

			// Act
			err := sut.SyncFlags(ctx, tt.flags)

			// Assert
			tt.assertResult(t, err)
		})
	}
}

func TestFliptWriter_CheckFlagExists(t *testing.T) {
	tests := []struct {
		name         string
		flagKey      string
		serverSetup  func() *httptest.Server
		assertResult func(t *testing.T, exists bool, err error)
	}{
		{
			name:    "Success_FlagExists",
			flagKey: "existing_flag",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusOK)
				}))
			},
			assertResult: func(t *testing.T, exists bool, err error) {
				assert.NoError(t, err)
				assert.True(t, exists)
			},
		},
		{
			name:    "Success_FlagNotFound",
			flagKey: "missing_flag",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusNotFound)
				}))
			},
			assertResult: func(t *testing.T, exists bool, err error) {
				assert.NoError(t, err)
				assert.False(t, exists)
			},
		},
		{
			name:    "Error_HTTPError",
			flagKey: "error_flag",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusInternalServerError)
				}))
			},
			assertResult: func(t *testing.T, exists bool, err error) {
				assert.NoError(t, err)
				assert.False(t, exists)
			},
		},
		{
			name:    "Error_ContextTimeout",
			flagKey: "timeout_flag",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					time.Sleep(100 * time.Millisecond)
					w.WriteHeader(http.StatusOK)
				}))
			},
			assertResult: func(t *testing.T, exists bool, err error) {
				assert.Error(t, err)
				assert.False(t, exists)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			server := tt.serverSetup()
			defer server.Close()

			sut := NewFliptWriter(server.URL, "default", &http.Client{})
			ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
			defer cancel()

			// Act
			exists, err := sut.checkFlagExists(ctx, tt.flagKey)

			// Assert
			tt.assertResult(t, exists, err)
		})
	}
}

func TestFliptWriter_CreateBooleanFlag(t *testing.T) {
	tests := []struct {
		name         string
		flagKey      string
		enabled      bool
		serverSetup  func() *httptest.Server
		assertResult func(t *testing.T, err error)
	}{
		{
			name:    "Success_FlagCreated",
			flagKey: "test_flag",
			enabled: true,
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusCreated)
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name:    "Error_HTTPError",
			flagKey: "error_flag",
			enabled: false,
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusBadRequest)
					w.Write([]byte("Invalid request"))
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "unexpected status code")
			},
		},
		{
			name:    "Error_ContextTimeout",
			flagKey: "timeout_flag",
			enabled: true,
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					time.Sleep(100 * time.Millisecond)
					w.WriteHeader(http.StatusCreated)
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.Error(t, err)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			server := tt.serverSetup()
			defer server.Close()

			sut := NewFliptWriter(server.URL, "default", &http.Client{})
			ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
			defer cancel()

			// Act
			err := sut.createBooleanFlag(ctx, tt.flagKey, tt.enabled)

			// Assert
			tt.assertResult(t, err)
		})
	}
}

func TestFliptWriter_CreateVariantFlag(t *testing.T) {
	tests := []struct {
		name         string
		flagKey      string
		serverSetup  func() *httptest.Server
		assertResult func(t *testing.T, err error)
	}{
		{
			name:    "Success_FlagCreated",
			flagKey: "variant_flag",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusCreated)
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name:    "Error_HTTPError",
			flagKey: "error_flag",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusBadRequest)
					w.Write([]byte("Invalid request"))
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "unexpected status code")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			server := tt.serverSetup()
			defer server.Close()

			sut := NewFliptWriter(server.URL, "default", &http.Client{})
			ctx := context.Background()

			// Act
			err := sut.createVariantFlag(ctx, tt.flagKey)

			// Assert
			tt.assertResult(t, err)
		})
	}
}

func TestFliptWriter_CreateBooleanVariant(t *testing.T) {
	tests := []struct {
		name         string
		flagKey      string
		enabled      bool
		serverSetup  func() *httptest.Server
		assertResult func(t *testing.T, err error)
	}{
		{
			name:    "Success_EnabledVariant",
			flagKey: "test_flag",
			enabled: true,
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusCreated)
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name:    "Success_DisabledVariant",
			flagKey: "test_flag",
			enabled: false,
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusCreated)
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name:    "Error_HTTPError",
			flagKey: "error_flag",
			enabled: true,
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusBadRequest)
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "unexpected status code")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			server := tt.serverSetup()
			defer server.Close()

			sut := NewFliptWriter(server.URL, "default", &http.Client{})
			ctx := context.Background()

			// Act
			err := sut.createBooleanVariant(ctx, tt.flagKey, tt.enabled)

			// Assert
			tt.assertResult(t, err)
		})
	}
}

func TestFliptWriter_CreateStringVariant(t *testing.T) {
	tests := []struct {
		name         string
		flagKey      string
		value        string
		serverSetup  func() *httptest.Server
		assertResult func(t *testing.T, err error)
	}{
		{
			name:    "Success_VariantCreated",
			flagKey: "variant_flag",
			value:   "variant_a",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusCreated)
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name:    "Error_HTTPError",
			flagKey: "error_flag",
			value:   "variant_b",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusBadRequest)
					w.Write([]byte("Invalid request"))
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "unexpected status code")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			server := tt.serverSetup()
			defer server.Close()

			sut := NewFliptWriter(server.URL, "default", &http.Client{})
			ctx := context.Background()

			// Act
			err := sut.createStringVariant(ctx, tt.flagKey, tt.value)

			// Assert
			tt.assertResult(t, err)
		})
	}
}

func TestFliptWriter_DeleteFlag(t *testing.T) {
	tests := []struct {
		name         string
		flagKey      string
		serverSetup  func() *httptest.Server
		assertResult func(t *testing.T, err error)
	}{
		{
			name:    "Success_FlagDeleted",
			flagKey: "test_flag",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusOK)
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name:    "Success_FlagNotFound",
			flagKey: "missing_flag",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusNotFound)
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name:    "Error_HTTPError",
			flagKey: "error_flag",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusInternalServerError)
					w.Write([]byte("Server error"))
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "unexpected status code")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			server := tt.serverSetup()
			defer server.Close()

			sut := NewFliptWriter(server.URL, "default", &http.Client{})
			ctx := context.Background()

			// Act
			err := sut.DeleteFlag(ctx, tt.flagKey)

			// Assert
			tt.assertResult(t, err)
		})
	}
}

func TestFliptWriter_DeleteAllFlags(t *testing.T) {
	tests := []struct {
		name         string
		serverSetup  func() *httptest.Server
		assertResult func(t *testing.T, err error)
	}{
		{
			name: "Success_AllFlagsDeleted",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.Method == http.MethodGet {
						// List flags response
						response := map[string]interface{}{
							"flags": []map[string]interface{}{
								{"key": "flag1"},
								{"key": "flag2"},
							},
						}
						json.NewEncoder(w).Encode(response)
					} else if r.Method == http.MethodDelete {
						w.WriteHeader(http.StatusOK)
					}
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name: "Success_NoFlagsToDelete",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					response := map[string]interface{}{
						"flags": []map[string]interface{}{},
					}
					json.NewEncoder(w).Encode(response)
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name: "Error_ListFlagsFails",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusInternalServerError)
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "unexpected status code")
			},
		},
		{
			name: "Error_JSONDecodeError",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.Write([]byte("invalid json"))
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "failed to decode flags list")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			server := tt.serverSetup()
			defer server.Close()

			sut := NewFliptWriter(server.URL, "default", &http.Client{})
			ctx := context.Background()

			// Act
			err := sut.DeleteAllFlags(ctx)

			// Assert
			tt.assertResult(t, err)
		})
	}
}

func TestFliptWriter_EnsureBooleanFlagExists(t *testing.T) {
	tests := []struct {
		name         string
		flagKey      string
		enabled      bool
		serverSetup  func() *httptest.Server
		assertResult func(t *testing.T, err error)
	}{
		{
			name:    "Success_FlagAlreadyExists",
			flagKey: "existing_flag",
			enabled: true,
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.Method == http.MethodGet {
						w.WriteHeader(http.StatusOK)
					}
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name:    "Success_FlagCreated",
			flagKey: "new_flag",
			enabled: false,
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.Method == http.MethodGet {
						w.WriteHeader(http.StatusNotFound)
					} else if r.Method == http.MethodPost {
						w.WriteHeader(http.StatusCreated)
					}
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name:    "Error_CheckFlagFails",
			flagKey: "error_flag",
			enabled: true,
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusInternalServerError)
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "failed to create boolean flag")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			server := tt.serverSetup()
			defer server.Close()

			sut := NewFliptWriter(server.URL, "default", &http.Client{})
			ctx := context.Background()

			// Act
			err := sut.ensureBooleanFlagExists(ctx, tt.flagKey, tt.enabled)

			// Assert
			tt.assertResult(t, err)
		})
	}
}

func TestFliptWriter_EnsureVariantFlagExists(t *testing.T) {
	tests := []struct {
		name         string
		flagKey      string
		value        string
		serverSetup  func() *httptest.Server
		assertResult func(t *testing.T, err error)
	}{
		{
			name:    "Success_FlagAlreadyExists",
			flagKey: "existing_flag",
			value:   "variant_a",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.Method == http.MethodGet {
						w.WriteHeader(http.StatusOK)
					}
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name:    "Success_FlagCreated",
			flagKey: "new_flag",
			value:   "variant_b",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.Method == http.MethodGet {
						w.WriteHeader(http.StatusNotFound)
					} else if r.Method == http.MethodPost {
						w.WriteHeader(http.StatusCreated)
					}
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name:    "Error_CheckFlagFails",
			flagKey: "error_flag",
			value:   "variant_c",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusInternalServerError)
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "failed to create variant flag")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			server := tt.serverSetup()
			defer server.Close()

			sut := NewFliptWriter(server.URL, "default", &http.Client{})
			ctx := context.Background()

			// Act
			err := sut.ensureVariantFlagExists(ctx, tt.flagKey, tt.value)

			// Assert
			tt.assertResult(t, err)
		})
	}
}
