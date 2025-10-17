package config

type StreamConfig struct {
	TransportFormat   string `mapstructure:"transport_format"`
	WebsocketEndpoint string `mapstructure:"websocket_endpoint"`
}
