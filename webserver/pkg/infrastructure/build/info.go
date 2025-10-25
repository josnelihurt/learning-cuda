package build

import "os"

// Info contains build-time information
type Info struct {
	Version    string
	Branch     string
	BuildTime  string
	CommitHash string
}

// These variables can be set at build time using ldflags
var (
	version    = "1.0.0"
	branch     = "main"
	buildTime  = "unknown"
	commitHash = "unknown"
)

// NewBuildInfo creates a new Info instance
// It reads from build-time ldflags or environment variables
func NewBuildInfo() *Info {
	info := &Info{
		Version:    version,
		Branch:     branch,
		BuildTime:  buildTime,
		CommitHash: commitHash,
	}

	// Override with environment variables if present
	if v := os.Getenv("APP_VERSION"); v != "" {
		info.Version = v
	}
	if b := os.Getenv("APP_BRANCH"); b != "" {
		info.Branch = b
	}
	if t := os.Getenv("BUILD_TIME"); t != "" {
		info.BuildTime = t
	}
	if c := os.Getenv("COMMIT_HASH"); c != "" {
		info.CommitHash = c
	}

	return info
}
