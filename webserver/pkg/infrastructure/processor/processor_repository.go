package processor

import (
	"github.com/jrb/cuda-learning/webserver/pkg/domain/interfaces"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/processor/loader"
)

const (
	// UnknownValue is the default value returned when processor info is not available
	UnknownValue = "unknown"
)

// RepositoryImpl implements interfaces.ProcessorRepository
type RepositoryImpl struct {
	registry *loader.Registry
}

// NewProcessorRepository creates a new processor repository
func NewProcessorRepository(registry *loader.Registry) interfaces.ProcessorRepository {
	return &RepositoryImpl{
		registry: registry,
	}
}

func (r *RepositoryImpl) GetAvailableLibraries() []string {
	if r.registry == nil {
		return []string{}
	}
	return r.registry.ListVersions()
}

func (r *RepositoryImpl) GetCurrentLibrary() string {
	if r.registry == nil {
		return UnknownValue
	}

	// Try to get the latest library as current
	if libInfo, err := r.registry.GetLatest(); err == nil {
		return libInfo.Metadata.Version
	}

	return UnknownValue
}

func (r *RepositoryImpl) GetAPIVersion() string {
	if r.registry == nil {
		return UnknownValue
	}

	// Try to get API version from latest library
	if libInfo, err := r.registry.GetLatest(); err == nil {
		if libInfo.Metadata.APIVersion != "" {
			return libInfo.Metadata.APIVersion
		}
	}

	return UnknownValue
}

func (r *RepositoryImpl) GetLibraryVersion() string {
	if r.registry == nil {
		return UnknownValue
	}

	// Get the latest library and call GetLibraryVersion on its loader
	libInfo, err := r.registry.GetLatest()
	if err != nil {
		return UnknownValue
	}

	// Load the library if not already loaded
	if libInfo.Loader == nil {
		loader, loadErr := r.registry.LoadLibrary(libInfo.Metadata.Version)
		if loadErr != nil {
			return UnknownValue
		}
		libInfo.Loader = loader
	}

	// Call GetLibraryVersion from C++
	version, err := libInfo.Loader.GetLibraryVersion()
	if err != nil {
		return UnknownValue
	}

	return version
}
