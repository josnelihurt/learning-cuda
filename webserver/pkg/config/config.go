package config

import (
	"context"
	"log"
	"time"

	"github.com/spf13/viper"
)

type Manager struct {
	HttpClientTimeout time.Duration
	FliptConfig       FliptConfig
	ServerConfig
	StreamConfig
	ObservabilityConfig
}

type FliptConfig struct {
	URL       string
	Namespace string
	DBPath    string
}

func (m *Manager) IsObservabilityEnabled(ctx context.Context) bool {
	return m.ObservabilityConfig.enabled
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
	viper.SetDefault("flipt.url", "http://localhost:8081")
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
	return &Manager{
		HttpClientTimeout: viper.GetDuration("flipt.http_timeout"),
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
		StreamConfig: StreamConfig{
			TransportFormat:   viper.GetString("stream.transport_format"),
			WebsocketEndpoint: viper.GetString("stream.websocket_endpoint"),
		},
		ObservabilityConfig: ObservabilityConfig{
			enabled:               viper.GetBool("observability.enabled"),
			ServiceName:           viper.GetString("observability.service_name"),
			ServiceVersion:        viper.GetString("observability.service_version"),
			OtelCollectorEndpoint: viper.GetString("observability.otel_collector_endpoint"),
			TraceSamplingRate:     viper.GetFloat64("observability.trace_sampling_rate"),
		},
		FliptConfig: FliptConfig{
			URL:       viper.GetString("flipt.url"),
			Namespace: viper.GetString("flipt.namespace"),
			DBPath:    viper.GetString("flipt.db_path"),
		},
	}
}
