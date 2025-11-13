package interfaces

// VersionRepository defines the interface for accessing version information from VERSION files
type VersionRepository interface {
	GetGoVersion() string
	GetCppVersion() string
	GetProtoVersion() string
}
