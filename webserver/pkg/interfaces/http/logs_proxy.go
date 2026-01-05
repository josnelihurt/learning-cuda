package http

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/logger"
)

type LogsProxyHandler struct {
	enabled           bool
	collectorEndpoint string
}

func NewLogsProxyHandler(collectorEndpoint string, enabled bool) *LogsProxyHandler {
	return &LogsProxyHandler{
		enabled:           enabled,
		collectorEndpoint: collectorEndpoint,
	}
}

func (h *LogsProxyHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Content-Encoding")

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	if !h.enabled {
		w.WriteHeader(http.StatusOK)
		if _, err := w.Write([]byte(`{"success":true,"message":"logging disabled"}`)); err != nil {
			log.Printf("Error writing response: %v", err)
		}
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("Error reading logs request body: %v", err)
		http.Error(w, "Failed to read body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	logger.Global().Debug().
		Int("bytes", len(body)).
		Str("content_type", r.Header.Get("Content-Type")).
		Msg("Received OTLP logs from frontend")

	if h.enabled && h.collectorEndpoint != "" {
		var collectorURL string
		if strings.HasPrefix(h.collectorEndpoint, "http://") || strings.HasPrefix(h.collectorEndpoint, "https://") {
			collectorURL = h.collectorEndpoint
		} else {
			collectorURL = fmt.Sprintf("http://%s/v1/logs", h.collectorEndpoint)
		}
		//nolint:gosec // collectorURL is from internal config, not user input
		resp, err := http.Post(collectorURL, "application/json", bytes.NewReader(body))
		if err != nil {
			logger.Global().Error().Err(err).Str("collector_url", collectorURL).Msg("Failed to forward logs to collector")
		} else {
			defer resp.Body.Close()
			if resp.StatusCode >= 400 {
				logger.Global().Warn().
					Int("status_code", resp.StatusCode).
					Str("collector_url", collectorURL).
					Msg("Collector returned error status")
			}
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write([]byte(`{"partialSuccess":{}}`)); err != nil {
		log.Printf("Error writing response: %v", err)
	}
}
