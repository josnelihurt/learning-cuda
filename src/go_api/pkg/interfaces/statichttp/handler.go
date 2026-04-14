package statichttp

import (
	"net/http"
	"path/filepath"
	"strings"

	"github.com/jrb/cuda-learning/src/go_api/pkg/application"
	"github.com/jrb/cuda-learning/src/go_api/pkg/config"
	"github.com/jrb/cuda-learning/src/go_api/pkg/domain"
	"github.com/jrb/cuda-learning/src/go_api/pkg/interfaces/websocket"
)

// StaticHandlerDeps groups dependencies for auxiliary HTTP routes (data files and WebSocket).
type StaticHandlerDeps struct {
	StreamConfig  config.StreamConfig
	UseCase       *application.ProcessImageUseCase
	VideoRepo     domain.VideoRepository
	EvaluateFFUC  *application.EvaluateFeatureFlagUseCase
	GRPCProcessor domain.ImageProcessor
}

type StaticHandler struct {
	wsHandler *websocket.Handler
}

func NewStaticHandler(deps *StaticHandlerDeps) *StaticHandler {
	return &StaticHandler{
		wsHandler: websocket.NewHandler(deps.UseCase, deps.StreamConfig, deps.VideoRepo, deps.EvaluateFFUC, deps.GRPCProcessor),
	}
}

func (h *StaticHandler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/data/", h.ServeData)
	mux.HandleFunc("/ws", h.wsHandler.HandleWebSocket)
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
