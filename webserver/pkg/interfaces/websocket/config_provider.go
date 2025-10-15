package websocket

type StreamConfigProvider interface {
	GetTransportFormat() string
	GetWebsocketEndpoint() string
}

