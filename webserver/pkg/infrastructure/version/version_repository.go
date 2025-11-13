package version

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/jrb/cuda-learning/webserver/pkg/domain/interfaces"
)

const (
	UnknownValue = "unknown"
)

type RepositoryImpl struct {
	goVersionPath    string
	cppVersionPath   string
	protoVersionPath string
}

func findProjectRoot() string {
	wd, err := os.Getwd()
	if err != nil {
		return ""
	}

	dir := wd
	for {
		webserverVersionPath := filepath.Join(dir, "webserver", "VERSION")
		if _, err := os.Stat(webserverVersionPath); err == nil {
			return dir
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}

	return wd
}

func NewVersionRepository() interfaces.VersionRepository {
	projectRoot := findProjectRoot()
	return &RepositoryImpl{
		goVersionPath:    filepath.Join(projectRoot, "webserver", "VERSION"),
		cppVersionPath:   filepath.Join(projectRoot, "cpp_accelerator", "VERSION"),
		protoVersionPath: filepath.Join(projectRoot, "proto", "VERSION"),
	}
}

func (r *RepositoryImpl) readVersionFile(filePath string) string {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return UnknownValue
	}
	return strings.TrimSpace(string(data))
}

func (r *RepositoryImpl) GetGoVersion() string {
	return r.readVersionFile(r.goVersionPath)
}

func (r *RepositoryImpl) GetCppVersion() string {
	return r.readVersionFile(r.cppVersionPath)
}

func (r *RepositoryImpl) GetProtoVersion() string {
	return r.readVersionFile(r.protoVersionPath)
}
