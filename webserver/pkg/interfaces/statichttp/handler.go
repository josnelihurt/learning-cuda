package statichttp

import (
	"html/template"
	"net/http"
	"path/filepath"
	"strings"

	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"github.com/jrb/cuda-learning/webserver/pkg/config"
	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"github.com/jrb/cuda-learning/webserver/pkg/interfaces/websocket"
)

type StaticHandler struct {
	webRootPath      string
	hotReloadEnabled bool
	tmpl             *template.Template
	wsHandler        *websocket.Handler
	assetHandler     AssetHandler
}

func NewStaticHandler(
	serverConfig *config.ServerConfig,
	streamConfig config.StreamConfig,
	useCase *application.ProcessImageUseCase,
	videoRepo domain.VideoRepository,
) *StaticHandler {
	var tmpl *template.Template
	if !serverConfig.HotReloadEnabled {
		templatePath := filepath.Join(serverConfig.WebRootPath, "templates", "index.html")
		tmpl = template.Must(template.ParseFiles(templatePath))
	}

	var assetHandler AssetHandler
	if serverConfig.HotReloadEnabled {
		assetHandler = NewDevelopmentAssetHandler(
			serverConfig.DevServerURL,
			serverConfig.DevServerPaths,
		)
	} else {
		assetHandler = NewProductionAssetHandler(serverConfig.WebRootPath)
	}

	return &StaticHandler{
		webRootPath:      serverConfig.WebRootPath,
		hotReloadEnabled: serverConfig.HotReloadEnabled,
		tmpl:             tmpl,
		wsHandler:        websocket.NewHandler(useCase, streamConfig, videoRepo),
		assetHandler:     assetHandler,
	}
}

func (h *StaticHandler) RegisterRoutes(mux *http.ServeMux) {
	if devHandler, ok := h.assetHandler.(*DevelopmentAssetHandler); ok {
		for _, prefix := range devHandler.GetPathPrefixes() {
			mux.HandleFunc(prefix, h.ServeAsset)
		}
	}

	mux.HandleFunc("/", h.ServeIndex)
	mux.HandleFunc("/static/", h.ServeStatic)
	mux.HandleFunc("/data/", h.ServeData)
	mux.HandleFunc("/ws", h.wsHandler.HandleWebSocket)
}

func (h *StaticHandler) ServeAsset(w http.ResponseWriter, r *http.Request) {
	h.assetHandler.ServeAsset(w, r)
}

func (h *StaticHandler) ServeIndex(w http.ResponseWriter, r *http.Request) {
	data := struct {
		ScriptTags []ScriptTag
	}{
		ScriptTags: h.assetHandler.GetScriptTags(),
	}

	tmpl := h.tmpl
	if h.hotReloadEnabled {
		templatePath := filepath.Join(h.webRootPath, "templates", "index.html")
		var err error
		tmpl, err = template.ParseFiles(templatePath)
		if err != nil {
			http.Error(w, "Template error", http.StatusInternalServerError)
			return
		}
	}

	if err := tmpl.Execute(w, data); err != nil {
		http.Error(w, "Template execution failed", http.StatusInternalServerError)
	}
}

func (h *StaticHandler) ServeStatic(w http.ResponseWriter, r *http.Request) {
	filePath := r.URL.Path[len("/static/"):]
	fullPath := filepath.Join(h.webRootPath, "static", filePath)

	if strings.HasSuffix(filePath, ".css") {
		w.Header().Set("Content-Type", "text/css")
	} else if strings.HasSuffix(filePath, ".js") {
		w.Header().Set("Content-Type", "application/javascript")
	}

	if !h.assetHandler.ShouldCacheAssets() {
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
		w.Header().Set("Pragma", "no-cache")
		w.Header().Set("Expires", "0")
	}

	http.ServeFile(w, r, fullPath)
}

func (h *StaticHandler) ServeData(w http.ResponseWriter, r *http.Request) {
	filePath := r.URL.Path[len("/data/"):]
	fullPath := filepath.Join("data", filePath)

	if strings.HasSuffix(filePath, ".png") {
		w.Header().Set("Content-Type", "image/png")
	} else if strings.HasSuffix(filePath, ".jpg") || strings.HasSuffix(filePath, ".jpeg") {
		w.Header().Set("Content-Type", "image/jpeg")
	}

	http.ServeFile(w, r, fullPath)
}
