package container

import (
	"context"
	"log"
	"time"

	"github.com/jrb/cuda-learning/webserver/internal/config"
	httpinfra "github.com/jrb/cuda-learning/webserver/internal/infrastructure/http"
	flipt "go.flipt.io/flipt-client"
)

type Container struct {
	Config             *config.Manager
	FliptClient        config.FliptClientInterface
	HTTPClient         *httpinfra.ClientProxy
	FliptWriter        *config.FliptWriter
	FeatureFlagManager *config.FeatureFlagManager
}

func New(ctx context.Context) (*Container, error) {
	cfg := config.New()

	httpClient := httpinfra.NewInstrumentedClient(httpinfra.ClientConfig{
		Timeout:         cfg.HttpClientTimeout,
		MaxIdleConns:    100,
		IdleConnTimeout: 90 * time.Second,
	})

	var fliptClientProxy config.FliptClientInterface
	var featureFlagMgr *config.FeatureFlagManager

	fliptClient, err := flipt.NewClient(
		ctx,
		flipt.WithURL(cfg.FliptConfig.URL),
		flipt.WithNamespace(cfg.FliptConfig.Namespace),
	)
	if err != nil {
		log.Printf("Warning: Failed to initialize Flipt client: %v. Feature flags will be disabled.", err)
		log.Printf("Note: Make sure Flipt is running and accessible at %s", cfg.FliptConfig.URL)
		fliptClientProxy = nil
		featureFlagMgr = nil
	} else {
		fliptClientProxy = config.NewFliptClient(fliptClient)
		fliptWriter := config.NewFliptWriter(
			cfg.FliptConfig.URL,
			cfg.FliptConfig.Namespace,
			httpClient,
		)
		featureFlagMgr = config.NewFeatureFlagManager(
			fliptClientProxy,
			fliptWriter,
			cfg.StreamConfig.TransportFormat,
			cfg.IsObservabilityEnabled(ctx),
		)
	}

	return &Container{
		Config:             cfg,
		FliptClient:        fliptClientProxy,
		HTTPClient:         httpClient,
		FliptWriter:        nil,
		FeatureFlagManager: featureFlagMgr,
	}, nil
}

func (c *Container) Close(ctx context.Context) error {
	if c.FliptClient != nil {
		closeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		if err := c.FliptClient.Close(closeCtx); err != nil {
			log.Printf("Error closing Flipt client: %v", err)
			return err
		}
	}
	return nil
}
