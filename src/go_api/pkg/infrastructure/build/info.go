package build

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
// It reads from build-time ldflags
func NewBuildInfo() *Info {
	return &Info{
		Version:    version,
		Branch:     branch,
		BuildTime:  buildTime,
		CommitHash: commitHash,
	}
}
