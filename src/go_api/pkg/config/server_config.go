package config

type ServerConfig struct {
	HTTPPort  string    `mapstructure:"http_port"`
	HTTPSPort string    `mapstructure:"https_port"`
	TLS       TLSConfig `mapstructure:"tls"`
}
