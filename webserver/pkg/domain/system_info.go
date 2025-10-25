package domain

// SystemVersion represents version information for different components
type SystemVersion struct {
	CppVersion string
	GoVersion  string
	JsVersion  string
	Branch     string
	BuildTime  string
	CommitHash string
}

// SystemInfo represents consolidated system information
type SystemInfo struct {
	Version            SystemVersion
	Environment        string
	CurrentLibrary     string
	APIVersion         string
	AvailableLibraries []string
}
