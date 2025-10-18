package statichttp

import (
	"crypto/tls"
	"fmt"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
)

type DevelopmentAssetHandler struct {
	devServerURL string
	proxy        *httputil.ReverseProxy
	pathPrefixes []string
}

func NewDevelopmentAssetHandler(devServerURL string, pathPrefixes []string) *DevelopmentAssetHandler {
	target, err := url.Parse(devServerURL)
	if err != nil {
		panic(fmt.Sprintf("Invalid dev server URL: %v", err))
	}
	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.Transport = &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
	}
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		log.Printf("Dev server proxy error: %v", err)
		http.Error(w, "Dev server unavailable", http.StatusBadGateway)
	}

	return &DevelopmentAssetHandler{
		devServerURL: devServerURL,
		proxy:        proxy,
		pathPrefixes: pathPrefixes,
	}
}

func (h *DevelopmentAssetHandler) ServeAsset(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
	w.Header().Set("Pragma", "no-cache")
	w.Header().Set("Expires", "0")

	h.proxy.ServeHTTP(w, r)
}

func (h *DevelopmentAssetHandler) GetScriptTags() []ScriptTag {
	return []ScriptTag{
		{Type: "module", Src: "/@vite/client", Module: true},
		{Type: "module", Src: "/src/main.ts", Module: true},
	}
}

func (h *DevelopmentAssetHandler) ShouldCacheAssets() bool {
	return false
}

func (h *DevelopmentAssetHandler) GetPathPrefixes() []string {
	return h.pathPrefixes
}
