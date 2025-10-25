package config

import (
	"github.com/jrb/cuda-learning/webserver/pkg/config"
	"github.com/jrb/cuda-learning/webserver/pkg/domain/interfaces"
)

// RepositoryImpl implements interfaces.ConfigRepository
type RepositoryImpl struct {
	configManager *config.Manager
}

// NewConfigRepository creates a new config repository
func NewConfigRepository(configManager *config.Manager) interfaces.ConfigRepository {
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

func (r *RepositoryImpl) GetDefaultLibrary() string {
	if r.configManager == nil {
		return "unknown"
	}
	return r.configManager.Processor.DefaultLibrary
}
