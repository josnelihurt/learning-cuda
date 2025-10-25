package statichttp

import (
	"io"
	"net/http"
	"strings"
)

// FliptProxyHandler handles proxying requests to Flipt with CSP headers removed
type FliptProxyHandler struct {
	fliptURL string
}

// NewFliptProxyHandler creates a new Flipt proxy handler
func NewFliptProxyHandler(fliptURL string) *FliptProxyHandler {
	return &FliptProxyHandler{
		fliptURL: fliptURL,
	}
}

// ServeHTTP proxies requests to Flipt and removes CSP headers
func (h *FliptProxyHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Construct the target URL
	targetURL := h.fliptURL + r.URL.Path
	if r.URL.RawQuery != "" {
		targetURL += "?" + r.URL.RawQuery
	}

	// Create a new request to Flipt
	req, err := http.NewRequest(r.Method, targetURL, r.Body)
	if err != nil {
		http.Error(w, "Failed to create request", http.StatusInternalServerError)
		return
	}

	// Copy headers from original request
	for name, values := range r.Header {
		// Skip host header to avoid conflicts
		if !strings.EqualFold(name, "host") {
			for _, value := range values {
				req.Header.Add(name, value)
			}
		}
	}

	// Set the host to the Flipt server
	req.Host = strings.TrimPrefix(h.fliptURL, "http://")
	req.Header.Set("Host", req.Host)

	// Make the request to Flipt
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		http.Error(w, "Failed to connect to Flipt", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// Copy response headers, but remove CSP headers
	for name, values := range resp.Header {
		// Remove CSP headers that prevent iframe embedding
		if strings.EqualFold(name, "content-security-policy") ||
			strings.EqualFold(name, "x-frame-options") {
			continue
		}
		for _, value := range values {
			w.Header().Add(name, value)
		}
	}

	// Set permissive headers for iframe embedding
	w.Header().Set("X-Frame-Options", "ALLOWALL")
	w.Header().Set("Content-Security-Policy", "frame-ancestors *")

	// Copy status code
	w.WriteHeader(resp.StatusCode)

	// Copy response body
	_, err = io.Copy(w, resp.Body)
	if err != nil {
		// Log error but don't fail the request since headers are already sent
		// This is a best-effort copy operation
		return
	}
}
