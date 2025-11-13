package interfaces

// ProcessorRepository defines the interface for accessing processor information
// Note: Currently unused after refactoring, kept for potential future use
type ProcessorRepository interface {
	GetLibraryVersion() string
}

// ConfigRepository defines the interface for accessing configuration
type ConfigRepository interface {
	GetEnvironment() string
}

// BuildInfoRepository defines the interface for accessing build information
type BuildInfoRepository interface {
	GetVersion() string
	GetBranch() string
	GetBuildTime() string
	GetCommitHash() string
}
