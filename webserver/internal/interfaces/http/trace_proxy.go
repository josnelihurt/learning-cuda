package http

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"net/http"
)

type TraceProxyHandler struct {
	collectorEndpoint string
	enabled           bool
}

func NewTraceProxyHandler(collectorEndpoint string, enabled bool) *TraceProxyHandler {
	return &TraceProxyHandler{
		collectorEndpoint: collectorEndpoint,
		enabled:           enabled,
	}
}

func (h *TraceProxyHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	if !h.enabled {
		http.Error(w, "Tracing disabled", http.StatusServiceUnavailable)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("Error reading trace request body: %v", err)
		http.Error(w, "Failed to read body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	collectorEndpoint := h.collectorEndpoint
	if collectorEndpoint == "localhost:4317" {
		collectorEndpoint = "localhost:4318"
	}
	collectorURL := fmt.Sprintf("http://%s/v1/traces", collectorEndpoint)
	log.Printf("Forwarding browser traces to: %s", collectorURL)
	
	proxyReq, err := http.NewRequest(http.MethodPost, collectorURL, bytes.NewReader(body))
	if err != nil {
		log.Printf("Error creating proxy request: %v", err)
		http.Error(w, "Failed to create proxy request", http.StatusInternalServerError)
		return
	}

	proxyReq.Header.Set("Content-Type", "application/json")
	for key, values := range r.Header {
		for _, value := range values {
			proxyReq.Header.Add(key, value)
		}
	}

	client := &http.Client{}
	resp, err := client.Do(proxyReq)
	if err != nil {
		log.Printf("Error forwarding traces to collector at %s: %v", collectorURL, err)
		http.Error(w, "Failed to forward traces", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	
	log.Printf("Traces forwarded successfully, collector responded with status: %d", resp.StatusCode)
	
	w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	
	w.WriteHeader(resp.StatusCode)
	w.Write(respBody)
}

