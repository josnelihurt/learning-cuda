package http

import (
	"encoding/json"
	"os"
	"path/filepath"
)

type ViteManifestEntry struct {
	File string `json:"file"`
	Src  string `json:"src"`
}

type ViteManifest map[string]ViteManifestEntry

func readViteManifest(webRootPath string) (ViteManifest, error) {
	manifestPath := filepath.Join(webRootPath, "static", "js", "dist", ".vite", "manifest.json")
	
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, err
	}
	
	var manifest ViteManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return nil, err
	}
	
	return manifest, nil
}

func getBundleFile(webRootPath string) string {
	manifest, err := readViteManifest(webRootPath)
	if err != nil {
		return "app.js"
	}
	
	if entry, ok := manifest["src/main.ts"]; ok {
		return entry.File
	}
	
	return "app.js"
}

