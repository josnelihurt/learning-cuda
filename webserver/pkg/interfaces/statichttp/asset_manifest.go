package statichttp

import (
	"encoding/json"
	"os"
	"path/filepath"
)

type AssetManifest interface {
	GetEntryFile(route string) string
}

type ViteManifestEntry struct {
	File string `json:"file"`
	Src  string `json:"src"`
}

type ViteManifest map[string]ViteManifestEntry

func (m ViteManifest) GetEntryFile(route string) string {
	key := "templates/index.html"
	if route == "react" {
		key = "templates/react.html"
	}
	if entry, ok := m[key]; ok {
		return entry.File
	}
	return "app.js"
}

func loadAssetManifest(webRootPath string) AssetManifest {
	manifestPath := filepath.Join(webRootPath, "static", "js", "dist", ".vite", "manifest.json")

	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return ViteManifest{}
	}

	var manifest ViteManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return ViteManifest{}
	}

	return manifest
}
