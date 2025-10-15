package http

import (
	"net/http"
	"time"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

type ClientConfig struct {
	Timeout         time.Duration
	MaxIdleConns    int
	IdleConnTimeout time.Duration
}

func NewClient(config ClientConfig) *http.Client {
	return &http.Client{
		Timeout: config.Timeout,
		Transport: &http.Transport{
			MaxIdleConns:    config.MaxIdleConns,
			IdleConnTimeout: config.IdleConnTimeout,
		},
	}
}

func NewInstrumentedClient(config ClientConfig) *ClientProxy {
	baseClient := NewClient(config)
	baseClient.Transport = otelhttp.NewTransport(baseClient.Transport)
	return &ClientProxy{client: baseClient}
}
