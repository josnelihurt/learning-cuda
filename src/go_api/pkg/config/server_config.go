package config

type ServerConfig struct {
	HTTPPort                string    `mapstructure:"http_port"`
	HTTPSPort               string    `mapstructure:"https_port"`
	WebRTCSignalingEndpoint string    `mapstructure:"webrtc_signaling_endpoint"`
	TLS                     TLSConfig `mapstructure:"tls"`
}
