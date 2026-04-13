package build

import (
	"github.com/jrb/cuda-learning/webserver/pkg/domain/interfaces"
)

const (
	// UnknownValue is the default value returned when build info is not available
	UnknownValue = "unknown"
)

// InfoRepositoryImpl implements interfaces.BuildInfoRepository
type InfoRepositoryImpl struct {
	buildInfo *Info
}

// NewBuildInfoRepository creates a new build info repository
func NewBuildInfoRepository(buildInfo *Info) interfaces.BuildInfoRepository {
	return &InfoRepositoryImpl{
		buildInfo: buildInfo,
	}
}

func (r *InfoRepositoryImpl) GetVersion() string {
	if r.buildInfo == nil {
		return UnknownValue
	}
	return r.buildInfo.Version
}

func (r *InfoRepositoryImpl) GetBranch() string {
	if r.buildInfo == nil {
		return UnknownValue
	}
	return r.buildInfo.Branch
}

func (r *InfoRepositoryImpl) GetBuildTime() string {
	if r.buildInfo == nil {
		return UnknownValue
	}
	return r.buildInfo.BuildTime
}

func (r *InfoRepositoryImpl) GetCommitHash() string {
	if r.buildInfo == nil {
		return UnknownValue
	}
	return r.buildInfo.CommitHash
}
