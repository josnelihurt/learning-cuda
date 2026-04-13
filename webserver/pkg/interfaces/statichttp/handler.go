package statichttp

import (
	"html/template"
	"net/http"
	"path/filepath"
	"strings"

	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"github.com/jrb/cuda-learning/webserver/pkg/config"
	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/build"
	"github.com/jrb/cuda-learning/webserver/pkg/interfaces/websocket"
)

type StaticHandler struct {
	webRootPath      string
	hotReloadEnabled bool
	litTmpl          *template.Template
	reactTmpl        *template.Template
	wsHandler        *websocket.Handler
	assetHandler     AssetHandler
	fliptURL         string
}

// StaticHandlerDeps groups all dependencies needed to create a StaticHandler.
type StaticHandlerDeps struct {
	ServerConfig  *config.ServerConfig
	StreamConfig  config.StreamConfig
	UseCase       *application.ProcessImageUseCase
	VideoRepo     domain.VideoRepository
	FliptURL      string
	EvaluateFFUC  *application.EvaluateFeatureFlagUseCase
	GRPCProcessor domain.ImageProcessor
}

func NewStaticHandler(deps *StaticHandlerDeps) *StaticHandler {
	var litTmpl, reactTmpl *template.Template
	if !deps.ServerConfig.HotReloadEnabled {
		litTemplatePath := filepath.Join(deps.ServerConfig.WebRootPath, "templates", "index.html")
		reactTemplatePath := filepath.Join(deps.ServerConfig.WebRootPath, "templates", "react.html")
		litTmpl = template.Must(template.ParseFiles(litTemplatePath))
		reactTmpl = template.Must(template.ParseFiles(reactTemplatePath))
	}

	var assetHandler AssetHandler
	if deps.ServerConfig.HotReloadEnabled {
		assetHandler = NewDevelopmentAssetHandler(
			deps.ServerConfig.DevServerURL,
			deps.ServerConfig.DevServerPaths,
		)
	} else {
		assetHandler = NewProductionAssetHandler(deps.ServerConfig.WebRootPath)
	}

	return &StaticHandler{
		webRootPath:      deps.ServerConfig.WebRootPath,
		hotReloadEnabled: deps.ServerConfig.HotReloadEnabled,
		litTmpl:          litTmpl,
		reactTmpl:        reactTmpl,
		wsHandler:        websocket.NewHandler(deps.UseCase, deps.StreamConfig, deps.VideoRepo, deps.EvaluateFFUC, deps.GRPCProcessor),
		assetHandler:     assetHandler,
		fliptURL:         deps.FliptURL,
	}
}

func (h *StaticHandler) RegisterRoutes(mux *http.ServeMux) {
	if devHandler, ok := h.assetHandler.(*DevelopmentAssetHandler); ok {
		for _, prefix := range devHandler.GetPathPrefixes() {
			mux.HandleFunc(prefix, h.ServeAsset)
		}
	}

	// Register specific routes first (more specific patterns)
	mux.HandleFunc("/static/", h.ServeStatic)
	mux.HandleFunc("/data/", h.ServeData)
	mux.HandleFunc("/ws", h.wsHandler.HandleWebSocket)

	// Register Flipt proxy route
	fliptProxy := NewFliptProxyHandler(h.fliptURL)
	mux.Handle("/flipt/", http.StripPrefix("/flipt", fliptProxy))

	// Note: catch-all route "/" is not registered here to allow Vanguard to handle it first
	// If Vanguard returns 404, it means we should serve the SPA index
}

// GetServeIndex returns the ServeIndex handler for use as a fallback
func (h *StaticHandler) GetServeIndex() http.HandlerFunc {
	return h.ServeIndex
}

func (h *StaticHandler) ServeAsset(w http.ResponseWriter, r *http.Request) {
	h.assetHandler.ServeAsset(w, r)
}

func (h *StaticHandler) ServeIndex(w http.ResponseWriter, r *http.Request) {
	// Don't serve index.html for API routes
	if strings.HasPrefix(r.URL.Path, "/api/") {
		http.NotFound(w, r)
		return
	}

	// Determine which frontend to serve based on URL path
	route := "lit"
	if strings.HasPrefix(r.URL.Path, "/react") {
		route = "react"
	}

	// Get build information
	buildInfo := build.NewBuildInfo()

	data := struct {
		ScriptTags []ScriptTag
		CommitHash string
		Version    string
		Branch     string
		BuildTime  string
	}{
		ScriptTags: h.assetHandler.GetScriptTags(route),
		CommitHash: buildInfo.CommitHash,
		Version:    buildInfo.Version,
		Branch:     buildInfo.Branch,
		BuildTime:  buildInfo.BuildTime,
	}

	// Select template based on route
	var tmpl *template.Template
	if h.hotReloadEnabled {
		templateName := "index.html"
		if route == "react" {
			templateName = "react.html"
		}
		templatePath := filepath.Join(h.webRootPath, "templates", templateName)
		var err error
		tmpl, err = template.ParseFiles(templatePath)
		if err != nil {
			http.Error(w, "Template error", http.StatusInternalServerError)
			return
		}
	} else {
		if route == "react" {
			tmpl = h.reactTmpl
		} else {
			tmpl = h.litTmpl
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
		w.Header().Set("Content-Type", "application/javascript; charset=utf-8")
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
