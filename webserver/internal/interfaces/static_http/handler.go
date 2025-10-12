package static_http

import (
	"html/template"
	"net/http"
	"path/filepath"
	"strings"

	"github.com/jrb/cuda-learning/webserver/internal/config"
	"github.com/jrb/cuda-learning/webserver/internal/interfaces/connectrpc"
)

type StaticHandler struct {
	webRootPath      string
	hotReloadEnabled bool
	tmpl             *template.Template
	wsHandler        *WebSocketHandler
}

func NewStaticHandler(cfg *config.Config, rpcHandler *connectrpc.ImageProcessorHandler) *StaticHandler {
	var tmpl *template.Template
	if !cfg.Server.HotReloadEnabled {
		templatePath := filepath.Join(cfg.Server.WebRootPath, "templates", "index.html")
		tmpl = template.Must(template.ParseFiles(templatePath))
	}
	
	return &StaticHandler{
		webRootPath:      cfg.Server.WebRootPath,
		hotReloadEnabled: cfg.Server.HotReloadEnabled,
		tmpl:             tmpl,
		wsHandler:        NewWebSocketHandler(rpcHandler, cfg),
	}
}

func (h *StaticHandler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/", h.ServeIndex)
	mux.HandleFunc("/static/", h.ServeStatic)
	mux.HandleFunc("/data/", h.ServeData)
	mux.HandleFunc("/ws", h.wsHandler.HandleWebSocket)
}

func (h *StaticHandler) ServeIndex(w http.ResponseWriter, r *http.Request) {
	data := struct {
		DevMode    bool
		BundleFile string
	}{
		DevMode:    h.hotReloadEnabled,
		BundleFile: getBundleFile(h.webRootPath),
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
	
	tmpl.Execute(w, data)
}

func (h *StaticHandler) ServeStatic(w http.ResponseWriter, r *http.Request) {
	filePath := r.URL.Path[len("/static/"):]
	fullPath := filepath.Join(h.webRootPath, "static", filePath)
	
	if strings.HasSuffix(filePath, ".css") {
		w.Header().Set("Content-Type", "text/css")
	} else if strings.HasSuffix(filePath, ".js") {
		w.Header().Set("Content-Type", "application/javascript")
	}
	
	if h.hotReloadEnabled {
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

