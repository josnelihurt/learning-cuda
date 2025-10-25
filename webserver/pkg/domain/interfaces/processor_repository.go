package interfaces

// ProcessorRepository defines the interface for accessing processor information
type ProcessorRepository interface {
	GetAvailableLibraries() []string
	GetCurrentLibrary() string
	GetAPIVersion() string
}

// ConfigRepository defines the interface for accessing configuration
type ConfigRepository interface {
	GetEnvironment() string
	GetDefaultLibrary() string
}

// BuildInfoRepository defines the interface for accessing build information
type BuildInfoRepository interface {
	GetVersion() string
	GetBranch() string
	GetBuildTime() string
	GetCommitHash() string
}
