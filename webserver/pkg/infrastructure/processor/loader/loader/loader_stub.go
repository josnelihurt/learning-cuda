package loader

import (
	"fmt"

	pb "github.com/jrb/cuda-learning/proto/gen"
)

const CurrentAPIVersion = "2.0.0"

type Loader struct {
	Path         string
	apiVersion   string
	capabilities *pb.LibraryCapabilities
}

type Registry struct{}

func NewLoader(libraryPath string) (*Loader, error) {
	return nil, fmt.Errorf("loader not available during testing")
}

func NewRegistry(libraryBasePath string) *Registry {
	return &Registry{}
}

func (r *Registry) ListVersions() []string {
	return []string{}
}

func (r *Registry) GetAllLibraries() map[string]int {
	return map[string]int{}
}

func (r *Registry) LoadLibrary(version string) (*Loader, error) {
	return nil, fmt.Errorf("loader not available during testing")
}

func (r *Registry) Discover() error {
	return fmt.Errorf("loader not available during testing")
}

func (r *Registry) GetByVersion(version string) (*Loader, error) {
	return nil, fmt.Errorf("loader not available during testing")
}

func (l *Loader) Init(req *pb.InitRequest) (*pb.InitResponse, error) {
	return nil, fmt.Errorf("loader not available during testing")
}

func (l *Loader) ProcessImage(req *pb.ProcessImageRequest) (*pb.ProcessImageResponse, error) {
	return nil, fmt.Errorf("loader not available during testing")
}

func (l *Loader) GetCapabilities(req *pb.GetCapabilitiesRequest) (*pb.GetCapabilitiesResponse, error) {
	return nil, fmt.Errorf("loader not available during testing")
}

func (l *Loader) Cleanup() {}

func (l *Loader) CachedCapabilities() *pb.LibraryCapabilities {
	return l.capabilities
}

func (l *Loader) GetVersion() string {
	if l.apiVersion != "" {
		return l.apiVersion
	}
	return "stub"
}

func (l *Loader) IsCompatibleWith(apiVersion string) bool {
	return isCompatible(l.apiVersion, apiVersion)
}

// Helper functions for testing
func isCompatible(v1, v2 string) bool {
	major1 := getMajorVersion(v1)
	major2 := getMajorVersion(v2)
	return major1 == major2 && major1 > 0
}

func getMajorVersion(version string) int {
	if version == "" {
		return 0
	}

	for i, char := range version {
		if char == '.' {
			if i > 0 {
				// Parse the major version part
				majorStr := version[:i]
				switch majorStr {
				case "1":
					return 1
				case "2":
					return 2
				case "3":
					return 3
				}
			}
			return 0
		}
		if char < '0' || char > '9' {
			return 0
		}
	}

	// Single number version
	switch version {
	case "1":
		return 1
	case "2":
		return 2
	case "3":
		return 3
	}

	return 0
}
