package build

const (
	UnknownValue = "unknown"
)

type InfoRepositoryImpl struct {
	buildInfo *Info
}

func NewBuildInfoRepository(buildInfo *Info) *InfoRepositoryImpl {
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
