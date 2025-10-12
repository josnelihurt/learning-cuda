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
	Port              string
	HotReloadEnabled  bool
	WebRootPath       string
}

type StreamConfig struct {
	TransportFormat   string
	WebsocketEndpoint string
}

func Load() *Config {
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath("./webserver")
	viper.AddConfigPath(".")

	viper.SetDefault("server.port", ":8080")
	viper.SetDefault("server.hot_reload_enabled", false)
	viper.SetDefault("server.web_root_path", "webserver/web")
	viper.SetDefault("stream.transport_format", "json")
	viper.SetDefault("stream.websocket_endpoint", "/ws")

	if err := viper.ReadInConfig(); err != nil {
		log.Printf("Warning: Config file not found, using defaults: %v", err)
	}

	cfg := &Config{
		Server: ServerConfig{
			Port:             viper.GetString("server.port"),
			HotReloadEnabled: viper.GetBool("server.hot_reload_enabled"),
			WebRootPath:      viper.GetString("server.web_root_path"),
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

