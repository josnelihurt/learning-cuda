package statichttp

import (
	"net/http"
	"path/filepath"
	"strings"
)

type ProductionAssetHandler struct {
	webRootPath string
	manifest    AssetManifest
}

func NewProductionAssetHandler(webRootPath string) *ProductionAssetHandler {
	return &ProductionAssetHandler{
		webRootPath: webRootPath,
		manifest:    loadAssetManifest(webRootPath),
	}
}

func (h *ProductionAssetHandler) ServeAsset(w http.ResponseWriter, r *http.Request) {
	filePath := r.URL.Path[len("/static/"):]
	fullPath := filepath.Join(h.webRootPath, "static", filePath)

	if strings.HasSuffix(filePath, ".css") {
		w.Header().Set("Content-Type", "text/css")
	} else if strings.HasSuffix(filePath, ".js") {
		w.Header().Set("Content-Type", "application/javascript")
	}

	http.ServeFile(w, r, fullPath)
}

func (h *ProductionAssetHandler) GetScriptTags() []ScriptTag {
	return []ScriptTag{
		{
			Type:   "module",
			Src:    "/static/js/dist/" + h.manifest.GetEntryFile(),
			Module: true,
		},
	}
}

func (h *ProductionAssetHandler) ShouldCacheAssets() bool {
	return true
}
