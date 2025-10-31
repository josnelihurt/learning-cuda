package loader

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

type LibraryMetadata struct {
	Name        string `json:"name"`
	Version     string `json:"version"`
	APIVersion  string `json:"api_version"`
	Type        string `json:"type"`
	BuildDate   string `json:"build_date"`
	BuildCommit string `json:"build_commit"`
	Description string `json:"description"`
}

type Registry struct {
	basePath  string
	libraries map[string]*LibraryInfo
}

type LibraryInfo struct {
	Metadata LibraryMetadata
	Path     string
	Loader   *Loader
}

func NewRegistry(basePath string) *Registry {
	return &Registry{
		basePath:  basePath,
		libraries: make(map[string]*LibraryInfo),
	}
}

func (r *Registry) Discover() error {
	pattern := regexp.MustCompile(`libcuda_processor_v(\d+\.\d+\.\d+)\.so$`)

	entries, err := os.ReadDir(r.basePath)
	if err != nil {
		return fmt.Errorf("failed to read library directory %s: %w", r.basePath, err)
	}

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".so") {
			continue
		}

		soPath := filepath.Join(r.basePath, entry.Name())

		metaPath := soPath + ".json"
		var metadata LibraryMetadata

		if data, err := os.ReadFile(metaPath); err == nil {
			if err := json.Unmarshal(data, &metadata); err != nil {
				continue
			}
		} else {
			matches := pattern.FindStringSubmatch(entry.Name())
			if len(matches) == 2 {
				metadata = LibraryMetadata{
					Version: matches[1],
					Type:    "gpu",
				}
			} else {
				continue
			}
		}

		r.libraries[metadata.Version] = &LibraryInfo{
			Metadata: metadata,
			Path:     soPath,
		}
	}

	return nil
}

func (r *Registry) LoadLibrary(version string) (*Loader, error) {
	info, ok := r.libraries[version]
	if !ok {
		return nil, fmt.Errorf("library version %s not found in registry", version)
	}

	if info.Loader == nil {
		loader, err := NewLoader(info.Path)
		if err != nil {
			return nil, fmt.Errorf("failed to load library %s: %w", info.Path, err)
		}
		info.Loader = loader
	}

	return info.Loader, nil
}

func (r *Registry) ListVersions() []string {
	versions := make([]string, 0, len(r.libraries))
	for v := range r.libraries {
		versions = append(versions, v)
	}
	sort.Strings(versions)
	return versions
}

func (r *Registry) GetLatest() (*LibraryInfo, error) {
	versions := r.GetVersions()
	if len(versions) == 0 {
		return nil, fmt.Errorf("no libraries found")
	}

	sort.Strings(versions)
	latestVersion := versions[len(versions)-1]
	return r.libraries[latestVersion], nil
}

func (r *Registry) GetByVersion(version string) (*LibraryInfo, error) {
	lib, ok := r.libraries[version]
	if !ok {
		return nil, fmt.Errorf("version %s not found", version)
	}
	return lib, nil
}

func (r *Registry) GetVersions() []string {
	versions := make([]string, 0, len(r.libraries))
	for v := range r.libraries {
		versions = append(versions, v)
	}
	return versions
}

func (r *Registry) GetAllLibraries() map[string]*LibraryInfo {
	return r.libraries
}
