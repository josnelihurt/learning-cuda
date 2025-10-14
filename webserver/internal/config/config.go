package config

import (
	"context"
	"log"
	"time"

	"github.com/spf13/viper"
)

type featureFlagsManager interface {
	Iterate(ctx context.Context, fn func(ctx context.Context, flagKey string, flagValue interface{}) error) error
}

// Manager is a configuration manager that implements domain-specific interfaces.
// It transparently proxies feature flags from Flipt to override YAML configuration values.
// When a feature flag is set for configuration overrides (e.g., stream.transport_format),
// the Manager returns the flag value; otherwise, it falls back to YAML values.
type Manager struct {
	appContext context.Context
	HttpClientTimeout time.Duration
	FliptConfig FliptConfig
	ServerConfig
	StreamConfig
	*ObservabilityConfig
}

func(m*Manager) Iterate(ctx context.Context, fn func(ctx context.Context, flagKey string, flagValue interface{}) error) error {
	flags := map[string]interface{}{
		"ws.transport_format": m.StreamConfig.TransportFormat,
		"observability.enabled": m.ObservabilityConfig.enabled,
	}
	for flagKey, flagValue := range flags {
		if err := fn(ctx, flagKey, flagValue); err != nil {
			return err
		}
	}
   return nil
}

func New() *Manager {
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath("./config")
	viper.AddConfigPath(".")

	viper.SetDefault("server.http_port", ":8080")
	viper.SetDefault("server.https_port", ":8443")
	viper.SetDefault("server.hot_reload_enabled", false)
	viper.SetDefault("server.web_root_path", "webserver/web")
	viper.SetDefault("server.dev_server_url", "https://localhost:3000")
	viper.SetDefault("server.dev_server_paths", []string{"/@vite/", "/src/", "/node_modules/"})
	viper.SetDefault("server.tls.enabled", true)
	viper.SetDefault("server.tls.cert_file", ".secrets/localhost+2.pem")
	viper.SetDefault("server.tls.key_file", ".secrets/localhost+2-key.pem")
	viper.SetDefault("stream.transport_format", "json")
	viper.SetDefault("stream.websocket_endpoint", "/ws")
	viper.SetDefault("observability.service_name", "cuda-image-processor")
	viper.SetDefault("observability.service_version", "1.0.0")
	viper.SetDefault("observability.otel_collector_endpoint", "localhost:4317")
	viper.SetDefault("observability.trace_sampling_rate", 1.0)
	viper.SetDefault("observability.enabled", true)
	viper.SetDefault("flipt.enabled", true)
	viper.SetDefault("flipt.url", "http://localhost:9000")
	viper.SetDefault("flipt.namespace", "default")
	viper.SetDefault("flipt.db_path", ".ignore/storage/flipt/flipt.db")
	viper.SetDefault("flipt.client_timeout", "30s")
	viper.SetDefault("flipt.update_interval", "30s")
	viper.SetDefault("flipt.http_timeout", "10s")

	viper.AutomaticEnv()
	viper.SetEnvPrefix("CUDA_PROCESSOR")

	if err := viper.ReadInConfig(); err != nil {
		log.Printf("Warning: Config file not found, using defaults: %v", err)
	}

	streamConfig := &StreamConfig{
		TransportFormat:   viper.GetString("stream.transport_format"),
		WebsocketEndpoint: viper.GetString("stream.websocket_endpoint"),
	}

	result := &Manager{
		ServerConfig: ServerConfig{
			HTTPPort:         viper.GetString("server.http_port"),
				HTTPSPort:        viper.GetString("server.https_port"),
			HotReloadEnabled: viper.GetBool("server.hot_reload_enabled"),
			WebRootPath:      viper.GetString("server.web_root_path"),
			DevServerURL:     viper.GetString("server.dev_server_url"),
			DevServerPaths:   viper.GetStringSlice("server.dev_server_paths"),
			TLSConfig: TLSConfig{
				Enabled:  viper.GetBool("server.tls.enabled"),
				CertFile: viper.GetString("server.tls.cert_file"),
				KeyFile:  viper.GetString("server.tls.key_file"),
			},
		},
		StreamConfig: *streamConfig,
		ObservabilityConfig: &ObservabilityConfig{
			enabled:             viper.GetBool("observability.enabled"),
			ServiceName:           viper.GetString("observability.service_name"),
			ServiceVersion:        viper.GetString("observability.service_version"),
			OtelCollectorEndpoint: viper.GetString("observability.otel_collector_endpoint"),
			TraceSamplingRate:     viper.GetFloat64("observability.trace_sampling_rate"),
		},
		FliptConfig: FliptConfig{
			URL:            viper.GetString("flipt.url"),
			Namespace:      viper.GetString("flipt.namespace"),
			DBPath:         viper.GetString("flipt.db_path"),
		},
	}

	return result
}


func (m *Manager) Close() {
	if m.fliptClient != nil {
		ctx, cancel := context.WithTimeout(m.appContext, 5*time.Second)
		defer cancel()
		if err := m.fliptClient.Close(ctx); err != nil {
			log.Printf("Error closing Flipt client: %v", err)
		}
	}
}

