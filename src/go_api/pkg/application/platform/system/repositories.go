package system

import (
	"context"

	gen "github.com/jrb/cuda-learning/proto/gen"
)

type configRepository interface {
	GetEnvironment() string
}

type buildInfoRepository interface {
	GetVersion() string
	GetBranch() string
	GetBuildTime() string
	GetCommitHash() string
}

type versionRepository interface {
	GetGoVersion() string
	GetProtoVersion() string
}

type processorCapabilitiesRepository interface {
	GetCapabilities(ctx context.Context) (*gen.LibraryCapabilities, error)
}
