package config

import (
	"context"
	"log"
	"strings"
	"time"

	"github.com/spf13/viper"
)

type Manager struct {
	HttpClientTimeout time.Duration
	Environment       string
	FliptConfig       FliptConfig
	ServerConfig
	StreamConfig
	ObservabilityConfig
	LoggerConfig
	ProcessorConfig
	ToolsConfig ToolsConfig
}

type FliptConfig struct {
	URL       string
	Namespace string
	DBPath    string
}

type ToolsConfig struct {
	Observability []ToolDefinition
	Features      []ToolDefinition
	Testing       []ToolDefinition
}

type ToolDefinition struct {
	ID       string `mapstructure:"id"`
	Name     string `mapstructure:"name"`
	IconPath string `mapstructure:"icon_path"`
	Type     string `mapstructure:"type"`
	URLDev   string `mapstructure:"url_dev"`
	URLProd  string `mapstructure:"url_prod"`
	Action   string `mapstructure:"action"`
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
	viper.SetDefault("logging.level", "info")
	viper.SetDefault("logging.format", "json")
	viper.SetDefault("logging.output", "stdout")
	viper.SetDefault("logging.include_caller", true)
	viper.SetDefault("processor.library_base_path", ".ignore/lib/cuda_learning")
	viper.SetDefault("processor.default_library", "mock")
	viper.SetDefault("processor.enable_hot_reload", false)
	viper.SetDefault("processor.fallback_enabled", true)
	viper.SetDefault("processor.fallback_chain", []string{"1.0.0", "mock"})
	viper.SetDefault("environment", "development")

	viper.SetEnvPrefix("CUDA_PROCESSOR")
	viper.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	viper.AutomaticEnv()

	if err := viper.ReadInConfig(); err != nil {
		log.Printf("Warning: Config file not found, using defaults: %v", err)
	}

	var toolsConfig ToolsConfig
	if err := viper.UnmarshalKey("tools", &toolsConfig); err != nil {
		log.Printf("Warning: Failed to unmarshal tools config: %v", err)
	}

	return &Manager{
		HttpClientTimeout: viper.GetDuration("flipt.http_timeout"),
		Environment:       viper.GetString("environment"),
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
		LoggerConfig: LoggerConfig{
			Level:         viper.GetString("logging.level"),
			Format:        viper.GetString("logging.format"),
			Output:        viper.GetString("logging.output"),
			IncludeCaller: viper.GetBool("logging.include_caller"),
		},
		ProcessorConfig: ProcessorConfig{
			LibraryBasePath: viper.GetString("processor.library_base_path"),
			DefaultLibrary:  viper.GetString("processor.default_library"),
			EnableHotReload: viper.GetBool("processor.enable_hot_reload"),
			FallbackEnabled: viper.GetBool("processor.fallback_enabled"),
			FallbackChain:   viper.GetStringSlice("processor.fallback_chain"),
		},
		ToolsConfig: toolsConfig,
	}
}
