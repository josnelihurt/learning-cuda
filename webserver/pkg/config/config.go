package config

import (
	"context"
	"log"
	"time"

	"github.com/spf13/viper"
)

type Manager struct {
	HTTPClientTimeout time.Duration       `mapstructure:"http_client_timeout"`
	Environment       string              `mapstructure:"environment"`
	Flipt             FliptConfig         `mapstructure:"flipt"`
	Server            ServerConfig        `mapstructure:"server"`
	Stream            StreamConfig        `mapstructure:"stream"`
	Observability     ObservabilityConfig `mapstructure:"observability"`
	Logging           LoggerConfig        `mapstructure:"logging"`
	Processor         ProcessorConfig     `mapstructure:"processor"`
	Tools             ToolsConfig         `mapstructure:"tools"`
	StaticImages      StaticImagesConfig  `mapstructure:"static_images"`
}

type FliptConfig struct {
	Enabled        bool          `mapstructure:"enabled"`
	URL            string        `mapstructure:"url"`
	Namespace      string        `mapstructure:"namespace"`
	DBPath         string        `mapstructure:"db_path"`
	ClientTimeout  time.Duration `mapstructure:"client_timeout"`
	UpdateInterval time.Duration `mapstructure:"update_interval"`
	HTTPTimeout    time.Duration `mapstructure:"http_timeout"`
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

type StaticImagesConfig struct {
	Directory string `mapstructure:"directory"`
}

func (m *Manager) IsObservabilityEnabled(ctx context.Context) bool {
	return m.Observability.Enabled
}

func New(configFile string) *Manager {
	v := viper.New()

	if configFile == "" {
		configFile = "config/config.yaml"
	}

	v.SetConfigFile(configFile)
	v.SetConfigType("yaml")

	setDefaults(v)

	if err := v.ReadInConfig(); err != nil {
		log.Printf("Warning: Config file not found: %v, using defaults", err)
	}

	var cfg Manager
	if err := v.Unmarshal(&cfg); err != nil {
		log.Fatalf("Unable to unmarshal config: %v", err)
	}

	return &cfg
}

func setDefaults(v *viper.Viper) {
	defaults := map[string]interface{}{
		"environment":         "development",
		"http_client_timeout": "10s",

		"server.http_port":          ":8080",
		"server.https_port":         ":8443",
		"server.hot_reload_enabled": false,
		"server.web_root_path":      "webserver/web",
		"server.dev_server_url":     "https://localhost:3000",
		"server.dev_server_paths":   []string{"/@vite/", "/src/", "/node_modules/"},
		"server.tls.enabled":        true,
		"server.tls.cert_file":      ".secrets/localhost+2.pem",
		"server.tls.key_file":       ".secrets/localhost+2-key.pem",

		"stream.transport_format":   "json",
		"stream.websocket_endpoint": "/ws",

		"observability.enabled":                 true,
		"observability.service_name":            "cuda-image-processor",
		"observability.service_version":         "1.0.0",
		"observability.otel_collector_endpoint": "localhost:4317",
		"observability.trace_sampling_rate":     1.0,

		"flipt.enabled":         true,
		"flipt.url":             "http://localhost:8081",
		"flipt.namespace":       "default",
		"flipt.db_path":         ".ignore/storage/flipt/flipt.db",
		"flipt.client_timeout":  "30s",
		"flipt.update_interval": "30s",
		"flipt.http_timeout":    "10s",

		"logging.level":          "info",
		"logging.format":         "json",
		"logging.output":         "stdout",
		"logging.include_caller": true,

		"processor.library_base_path":      ".ignore/lib/cuda_learning",
		"processor.grpc_server_address":    "localhost:60061",
		"processor.use_grpc_for_processor": true,

		"static_images.directory": "/data/static_images",
	}

	for key, value := range defaults {
		v.SetDefault(key, value)
	}
}
