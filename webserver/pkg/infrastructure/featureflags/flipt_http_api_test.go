package featureflags

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Test data builders
func makeValidFlag(key, name, flagType string, enabled bool) Flag {
	return Flag{
		Key:         key,
		Name:        name,
		Type:        flagType,
		Enabled:     enabled,
		Description: "Test flag",
	}
}

func makeValidFlagsList() []Flag {
	return []Flag{
		makeValidFlag("flag1", "Flag 1", "BOOLEAN_FLAG_TYPE", true),
		makeValidFlag("flag2", "Flag 2", "VARIANT_FLAG_TYPE", false),
	}
}

func makeValidFlagsResponse(flags []Flag) map[string]interface{} {
	return map[string]interface{}{
		"flags": flags,
	}
}

// Tests
func TestNewFliptHTTPAPI(t *testing.T) {
	// Arrange
	grpcURL := "grpc://localhost:9000"
	namespace := "default"
	client := &http.Client{}

	// Act
	sut := NewFliptHTTPAPI(grpcURL, namespace, client)

	// Assert
	require.NotNil(t, sut)
	assert.Equal(t, grpcURL, sut.apiURL)
	assert.Equal(t, namespace, sut.namespace)
	assert.Equal(t, client, sut.client)
}

func TestFliptHTTPAPI_GetFlag(t *testing.T) {

	tests := []struct {
		name         string
		flagKey      string
		serverSetup  func() *httptest.Server
		assertResult func(t *testing.T, flag *Flag, err error)
	}{
		{
			name:    "Success_FlagFound",
			flagKey: "test_flag",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					flag := makeValidFlag("test_flag", "Test Flag", "BOOLEAN_FLAG_TYPE", true)
					json.NewEncoder(w).Encode(flag)
				}))
			},
			assertResult: func(t *testing.T, flag *Flag, err error) {
				assert.NoError(t, err)
				require.NotNil(t, flag)
				assert.Equal(t, "test_flag", flag.Key)
				assert.Equal(t, "Test Flag", flag.Name)
				assert.Equal(t, "BOOLEAN_FLAG_TYPE", flag.Type)
				assert.True(t, flag.Enabled)
			},
		},
		{
			name:    "Error_FlagNotFound",
			flagKey: "missing_flag",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusNotFound)
				}))
			},
			assertResult: func(t *testing.T, flag *Flag, err error) {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "flag not found")
				assert.Nil(t, flag)
			},
		},
		{
			name:    "Error_UnexpectedStatus",
			flagKey: "error_flag",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusInternalServerError)
					w.Write([]byte("Server error"))
				}))
			},
			assertResult: func(t *testing.T, flag *Flag, err error) {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "unexpected status code")
				assert.Nil(t, flag)
			},
		},
		{
			name:    "Error_JSONDecodeError",
			flagKey: "invalid_flag",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusOK)
					w.Write([]byte("invalid json"))
				}))
			},
			assertResult: func(t *testing.T, flag *Flag, err error) {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "failed to decode flag")
				assert.Nil(t, flag)
			},
		},
		{
			name:    "Error_ContextTimeout",
			flagKey: "timeout_flag",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					time.Sleep(100 * time.Millisecond)
					flag := makeValidFlag("timeout_flag", "Timeout Flag", "BOOLEAN_FLAG_TYPE", true)
					json.NewEncoder(w).Encode(flag)
				}))
			},
			assertResult: func(t *testing.T, flag *Flag, err error) {
				assert.Error(t, err)
				assert.Nil(t, flag)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			server := tt.serverSetup()
			defer server.Close()

			sut := NewFliptHTTPAPI(server.URL, "default", &http.Client{})
			ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
			defer cancel()

			// Act
			flag, err := sut.GetFlag(ctx, tt.flagKey)

			// Assert
			tt.assertResult(t, flag, err)
		})
	}
}

func TestFliptHTTPAPI_ListFlags(t *testing.T) {
	tests := []struct {
		name         string
		serverSetup  func() *httptest.Server
		assertResult func(t *testing.T, flags []Flag, err error)
	}{
		{
			name: "Success_MultipleFlags",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					flags := makeValidFlagsList()
					response := makeValidFlagsResponse(flags)
					json.NewEncoder(w).Encode(response)
				}))
			},
			assertResult: func(t *testing.T, flags []Flag, err error) {
				assert.NoError(t, err)
				require.Len(t, flags, 2)
				assert.Equal(t, "flag1", flags[0].Key)
				assert.Equal(t, "flag2", flags[1].Key)
			},
		},
		{
			name: "Success_EmptyFlagsList",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					response := makeValidFlagsResponse([]Flag{})
					json.NewEncoder(w).Encode(response)
				}))
			},
			assertResult: func(t *testing.T, flags []Flag, err error) {
				assert.NoError(t, err)
				assert.Len(t, flags, 0)
			},
		},
		{
			name: "Error_HTTPError",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusInternalServerError)
					w.Write([]byte("Server error"))
				}))
			},
			assertResult: func(t *testing.T, flags []Flag, err error) {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "unexpected status code")
				assert.Nil(t, flags)
			},
		},
		{
			name: "Error_JSONDecodeError",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusOK)
					w.Write([]byte("invalid json"))
				}))
			},
			assertResult: func(t *testing.T, flags []Flag, err error) {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "failed to decode flags")
				assert.Nil(t, flags)
			},
		},
		{
			name: "Error_ContextTimeout",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					time.Sleep(100 * time.Millisecond)
					flags := makeValidFlagsList()
					response := makeValidFlagsResponse(flags)
					json.NewEncoder(w).Encode(response)
				}))
			},
			assertResult: func(t *testing.T, flags []Flag, err error) {
				assert.Error(t, err)
				assert.Nil(t, flags)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			server := tt.serverSetup()
			defer server.Close()

			sut := NewFliptHTTPAPI(server.URL, "default", &http.Client{})
			ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
			defer cancel()

			// Act
			flags, err := sut.ListFlags(ctx)

			// Assert
			tt.assertResult(t, flags, err)
		})
	}
}

func TestFliptHTTPAPI_CleanAllFlags(t *testing.T) {
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
						flags := makeValidFlagsList()
						response := makeValidFlagsResponse(flags)
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
					response := makeValidFlagsResponse([]Flag{})
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
				assert.Contains(t, err.Error(), "failed to list flags")
			},
		},
		{
			name: "Error_DeleteFlagFails",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.Method == http.MethodGet {
						flags := makeValidFlagsList()
						response := makeValidFlagsResponse(flags)
						json.NewEncoder(w).Encode(response)
					} else if r.Method == http.MethodDelete {
						w.WriteHeader(http.StatusInternalServerError)
					}
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "failed to delete flag")
			},
		},
		{
			name: "Error_JSONDecodeError",
			serverSetup: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusOK)
					w.Write([]byte("invalid json"))
				}))
			},
			assertResult: func(t *testing.T, err error) {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "failed to list flags")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			server := tt.serverSetup()
			defer server.Close()

			sut := NewFliptHTTPAPI(server.URL, "default", &http.Client{})
			ctx := context.Background()

			// Act
			err := sut.CleanAllFlags(ctx)

			// Assert
			tt.assertResult(t, err)
		})
	}
}

func TestFliptHTTPAPI_ContextCancellation(t *testing.T) {
	// Arrange
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	sut := NewFliptHTTPAPI(server.URL, "default", &http.Client{})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Act
	flag, err := sut.GetFlag(ctx, "test_flag")

	// Assert
	assert.Error(t, err)
	assert.Nil(t, flag)
}
