package config

type ServerConfig struct {
	HTTPPort         string    `mapstructure:"http_port"`
	HTTPSPort        string    `mapstructure:"https_port"`
	HotReloadEnabled bool      `mapstructure:"hot_reload_enabled"`
	WebRootPath      string    `mapstructure:"web_root_path"`
	DevServerURL     string    `mapstructure:"dev_server_url"`
	DevServerPaths   []string  `mapstructure:"dev_server_paths"`
	TLS              TLSConfig `mapstructure:"tls"`
}
