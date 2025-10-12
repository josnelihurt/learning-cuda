package app

import (
	"github.com/jrb/cuda-learning/webserver/internal/config"
)

func LoadConfig() *config.Config {
	return config.Load()
}

