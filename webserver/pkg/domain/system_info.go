package domain

// SystemVersion represents version information for different components
type SystemVersion struct {
	GoVersion    string
	CppVersion   string
	ProtoVersion string
	Branch       string
	BuildTime    string
	CommitHash   string
}

// SystemInfo represents consolidated system information
type SystemInfo struct {
	Version     SystemVersion
	Environment string
}
