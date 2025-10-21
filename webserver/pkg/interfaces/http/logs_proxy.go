package http

import (
	"io"
	"log"
	"net/http"
)

type LogsProxyHandler struct {
	enabled bool
}

func NewLogsProxyHandler(collectorEndpoint string, enabled bool) *LogsProxyHandler {
	return &LogsProxyHandler{
		enabled: enabled,
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

	log.Printf("Received OTLP logs from frontend: %d bytes (Content-Type: %s)",
		len(body), r.Header.Get("Content-Type"))

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write([]byte(`{"partialSuccess":{}}`)); err != nil {
		log.Printf("Error writing response: %v", err)
	}
}
