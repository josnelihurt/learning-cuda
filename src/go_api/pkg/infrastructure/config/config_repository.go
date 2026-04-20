package config

import (
	"github.com/jrb/cuda-learning/src/go_api/pkg/config"
)

type RepositoryImpl struct {
	configManager *config.Manager
}

func NewConfigRepository(configManager *config.Manager) *RepositoryImpl {
	return &RepositoryImpl{
		configManager: configManager,
	}
}

func (r *RepositoryImpl) GetEnvironment() string {
	if r.configManager == nil {
		return "production"
	}
	return r.configManager.Environment
}
