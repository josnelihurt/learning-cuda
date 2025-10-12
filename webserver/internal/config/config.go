package config

import (
	"log"

	"github.com/spf13/viper"
)

type Config struct {
	Server       ServerConfig
	Stream       StreamConfig
	FeatureFlags map[string]bool
}

type ServerConfig struct {
	HttpPort         string
	HttpsPort        string
	HotReloadEnabled bool
	WebRootPath      string
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

func Load() *Config {
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath("./config")
	viper.AddConfigPath(".")

	viper.SetDefault("server.http_port", ":8080")
	viper.SetDefault("server.https_port", ":8443")
	viper.SetDefault("server.hot_reload_enabled", false)
	viper.SetDefault("server.web_root_path", "webserver/web")
	viper.SetDefault("server.tls.enabled", true)
	viper.SetDefault("server.tls.cert_file", ".secrets/localhost+2.pem")
	viper.SetDefault("server.tls.key_file", ".secrets/localhost+2-key.pem")
	viper.SetDefault("stream.transport_format", "json")
	viper.SetDefault("stream.websocket_endpoint", "/ws")

	if err := viper.ReadInConfig(); err != nil {
		log.Printf("Warning: Config file not found, using defaults: %v", err)
	}

	cfg := &Config{
		Server: ServerConfig{
			HttpPort:         viper.GetString("server.http_port"),
			HttpsPort:        viper.GetString("server.https_port"),
			HotReloadEnabled: viper.GetBool("server.hot_reload_enabled"),
			WebRootPath:      viper.GetString("server.web_root_path"),
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

