package config

import (
	"context"
	"log"
	"time"

	"github.com/spf13/viper"
	flipt "go.flipt.io/flipt-client"
)

type Config struct {
	Server        ServerConfig
	Stream        StreamConfig
	Observability ObservabilityConfig
	Flipt         FliptConfig
	FeatureFlags  map[string]bool
	fliptClient   *flipt.Client
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
	ServiceName           string
	ServiceVersion        string
	OtelCollectorEndpoint string
	TraceSamplingRate     float64
}

type FliptConfig struct {
	Enabled   bool
	URL       string
	Namespace string
	DBPath    string
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
	viper.SetDefault("flipt.enabled", true)
	viper.SetDefault("flipt.url", "http://localhost:9000")
	viper.SetDefault("flipt.namespace", "default")
	viper.SetDefault("flipt.db_path", ".ignore/storage/flipt/flipt.db")

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
		Flipt: FliptConfig{
			Enabled:   viper.GetBool("flipt.enabled"),
			URL:       viper.GetString("flipt.url"),
			Namespace: viper.GetString("flipt.namespace"),
			DBPath:    viper.GetString("flipt.db_path"),
		},
		FeatureFlags: make(map[string]bool),
	}

	features := viper.GetStringMap("features")
	for key, val := range features {
		if boolVal, ok := val.(bool); ok {
			cfg.FeatureFlags[key] = boolVal
		}
	}

	if cfg.Flipt.Enabled {
		cfg.initFliptClient()
		cfg.syncYAMLToFlipt()
	}

	return cfg
}

func (c *Config) initFliptClient() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	client, err := flipt.NewClient(
		ctx,
		flipt.WithURL(c.Flipt.URL),
		flipt.WithNamespace(c.Flipt.Namespace),
		flipt.WithUpdateInterval(30*time.Second),
		flipt.WithFetchMode(flipt.FetchModePolling),
		flipt.WithErrorStrategy(flipt.ErrorStrategyFallback),
	)

	if err != nil {
		log.Printf("Failed to initialize Flipt SDK client: %v", err)
		return
	}

	c.fliptClient = client
}

func (c *Config) syncYAMLToFlipt() {
	if len(c.FeatureFlags) == 0 {
		return
	}
	
	writer := NewFliptWriter(c.Flipt.URL, c.Flipt.Namespace)
	if err := writer.SyncFlags(c.FeatureFlags); err != nil {
		log.Printf("Failed to sync flags to Flipt: %v", err)
	}
}

func (c *Config) GetFeature(name string) bool {
	if val, exists := c.FeatureFlags[name]; exists {
		return val
	}
	return false
}

func (c *Config) IsFeatureEnabled(name string) bool {
	if c.fliptClient != nil {
		enabled, err := c.evaluateFliptFlag(name)
		if err != nil {
			log.Printf("Flipt evaluation failed for '%s': %v. Using YAML fallback.", name, err)
			return c.GetFeature(name)
		}
		return enabled
	}

	return c.GetFeature(name)
}

func (c *Config) evaluateFliptFlag(key string) (bool, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	result, err := c.fliptClient.EvaluateBoolean(ctx, &flipt.EvaluationRequest{
		FlagKey:  key,
		EntityID: "system",
		Context:  map[string]string{},
	})

	if err != nil {
		return false, err
	}

	return result.Enabled, nil
}

func (c *Config) Close() {
	if c.fliptClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := c.fliptClient.Close(ctx); err != nil {
			log.Printf("Error closing Flipt client: %v", err)
		}
	}
}

