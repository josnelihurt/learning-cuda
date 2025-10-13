package config

import (
	"log"

	"github.com/spf13/viper"
)

type Config struct {
	Server       ServerConfig
	Stream       StreamConfig
	Observability ObservabilityConfig
	FeatureFlags map[string]bool
}

type ServerConfig struct {
	HttpPort         string
	HttpsPort        string
	HotReloadEnabled bool
	WebRootPath      string
	DevServerURL     string
	DevServerPaths   []string
	TLS              TLSConfig
}

type TLSConfig struct {
	Enabled  bool
	CertFile string
	KeyFile  string
}

type StreamConfig struct {
	TransportFormat   string
	WebsocketEndpoint string
}

type ObservabilityConfig struct {
	ServiceName            string
	ServiceVersion         string
	OtelCollectorEndpoint  string
	TraceSamplingRate      float64
}

func Load() *Config {
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

	viper.AutomaticEnv()
	viper.SetEnvPrefix("CUDA_PROCESSOR")

	if err := viper.ReadInConfig(); err != nil {
		log.Printf("Warning: Config file not found, using defaults: %v", err)
	}

	cfg := &Config{
		Server: ServerConfig{
			HttpPort:         viper.GetString("server.http_port"),
			HttpsPort:        viper.GetString("server.https_port"),
			HotReloadEnabled: viper.GetBool("server.hot_reload_enabled"),
			WebRootPath:      viper.GetString("server.web_root_path"),
			DevServerURL:     viper.GetString("server.dev_server_url"),
			DevServerPaths:   viper.GetStringSlice("server.dev_server_paths"),
			TLS: TLSConfig{
				Enabled:  viper.GetBool("server.tls.enabled"),
				CertFile: viper.GetString("server.tls.cert_file"),
				KeyFile:  viper.GetString("server.tls.key_file"),
			},
		},
		Stream: StreamConfig{
			TransportFormat:   viper.GetString("stream.transport_format"),
			WebsocketEndpoint: viper.GetString("stream.websocket_endpoint"),
		},
		Observability: ObservabilityConfig{
			ServiceName:           viper.GetString("observability.service_name"),
			ServiceVersion:        viper.GetString("observability.service_version"),
			OtelCollectorEndpoint: viper.GetString("observability.otel_collector_endpoint"),
			TraceSamplingRate:     viper.GetFloat64("observability.trace_sampling_rate"),
		},
		FeatureFlags: make(map[string]bool),
	}

	features := viper.GetStringMap("features")
	for key, val := range features {
		if boolVal, ok := val.(bool); ok {
			cfg.FeatureFlags[key] = boolVal
		}
	}

	return cfg
}

func (c *Config) GetFeature(name string) bool {
	if val, exists := c.FeatureFlags[name]; exists {
		return val
	}
	return false
}

func (c *Config) IsFeatureEnabled(name string) bool {
	return c.GetFeature(name)
}

