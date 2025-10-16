package container

import (
	"context"
	"time"

	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"github.com/jrb/cuda-learning/webserver/pkg/config"
	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/featureflags"
	httpinfra "github.com/jrb/cuda-learning/webserver/pkg/infrastructure/http"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/logger"
	flipt "go.flipt.io/flipt-client"
)

type Container struct {
	Config     *config.Manager
	HTTPClient *httpinfra.ClientProxy

	FeatureFlagRepo domain.FeatureFlagRepository

	ProcessImageUseCase        *application.ProcessImageUseCase
	EvaluateFeatureFlagUseCase *application.EvaluateFeatureFlagUseCase
	SyncFeatureFlagsUseCase    *application.SyncFeatureFlagsUseCase
	GetStreamConfigUseCase     *application.GetStreamConfigUseCase
	ListInputsUseCase          *application.ListInputsUseCase

	fliptClientProxy featureflags.FliptClientInterface
}

func New(ctx context.Context) (*Container, error) {
	cfg := config.New()

	log := logger.New(logger.Config{
		Level:         cfg.LoggerConfig.Level,
		Format:        cfg.LoggerConfig.Format,
		Output:        cfg.LoggerConfig.Output,
		IncludeCaller: cfg.LoggerConfig.IncludeCaller,
	})

	httpClient := httpinfra.NewInstrumentedClient(httpinfra.ClientConfig{
		Timeout:         cfg.HttpClientTimeout,
		MaxIdleConns:    100,
		IdleConnTimeout: 90 * time.Second,
	})

	var fliptClientProxy featureflags.FliptClientInterface
	var featureFlagRepo domain.FeatureFlagRepository

	fliptClient, err := flipt.NewClient(
		ctx,
		flipt.WithURL(cfg.FliptConfig.URL),
		flipt.WithNamespace(cfg.FliptConfig.Namespace),
	)
	if err != nil {
		log.Warn().
			Err(err).
			Str("flipt_url", cfg.FliptConfig.URL).
			Msg("Failed to initialize Flipt client. Feature flags will be disabled")
		fliptClientProxy = nil
		featureFlagRepo = nil
	} else {
		fliptClientProxy = featureflags.NewFliptClient(fliptClient)
		fliptWriter := featureflags.NewFliptWriter(
			cfg.FliptConfig.URL,
			cfg.FliptConfig.Namespace,
			httpClient,
		)
		featureFlagRepo = featureflags.NewFliptRepository(fliptClientProxy, fliptWriter)
	}

	evaluateFFUseCase := application.NewEvaluateFeatureFlagUseCase(featureFlagRepo)
	syncFFUseCase := application.NewSyncFeatureFlagsUseCase(featureFlagRepo)
	getStreamConfigUseCase := application.NewGetStreamConfigUseCase(evaluateFFUseCase, cfg.StreamConfig)
	listInputsUseCase := application.NewListInputsUseCase()

	return &Container{
		Config:                     cfg,
		HTTPClient:                 httpClient,
		FeatureFlagRepo:            featureFlagRepo,
		EvaluateFeatureFlagUseCase: evaluateFFUseCase,
		SyncFeatureFlagsUseCase:    syncFFUseCase,
		GetStreamConfigUseCase:     getStreamConfigUseCase,
		ListInputsUseCase:          listInputsUseCase,
		fliptClientProxy:           fliptClientProxy,
	}, nil
}

func (c *Container) Close(ctx context.Context) error {
	log := logger.Global()
	if c.fliptClientProxy != nil {
		closeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		if err := c.fliptClientProxy.Close(closeCtx); err != nil {
			log.Error().Err(err).Msg("Error closing Flipt client")
			return err
		}
	}
	return nil
}
