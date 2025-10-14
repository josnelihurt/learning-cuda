package static_http

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/jrb/cuda-learning/webserver/internal/config"
)

type FliptSyncHandler struct {
	config *config.Config
}

func NewFliptSyncHandler(cfg *config.Config) *FliptSyncHandler {
	return &FliptSyncHandler{
		config: cfg,
	}
}

type SyncResponse struct {
	Success bool              `json:"success"`
	Message string            `json:"message"`
	Flags   map[string]bool   `json:"flags,omitempty"`
	Errors  []string          `json:"errors,omitempty"`
}

func (h *FliptSyncHandler) HandleSyncFlags(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "application/json")

	if !h.config.Flipt.Enabled {
		json.NewEncoder(w).Encode(SyncResponse{
			Success: false,
			Message: "Flipt is not enabled in configuration",
		})
		return
	}

	if len(h.config.FeatureFlags) == 0 {
		json.NewEncoder(w).Encode(SyncResponse{
			Success: false,
			Message: "No feature flags defined in YAML configuration",
		})
		return
	}

	log.Printf("Manual flag sync triggered - syncing %d flags to Flipt", len(h.config.FeatureFlags))
	
	writer := config.NewFliptWriter(h.config.Flipt.URL, h.config.Flipt.Namespace)
	err := writer.SyncFlags(h.config.FeatureFlags)
	
	if err != nil {
		log.Printf("Flag sync failed: %v", err)
		json.NewEncoder(w).Encode(SyncResponse{
			Success: false,
			Message: "Failed to sync flags to Flipt",
			Errors:  []string{err.Error()},
		})
		return
	}

	log.Println("Manual flag sync completed successfully")
	json.NewEncoder(w).Encode(SyncResponse{
		Success: true,
		Message: "Flags synced successfully to Flipt",
		Flags:   h.config.FeatureFlags,
	})
}

