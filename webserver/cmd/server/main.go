package main

import (
	"flag"
	"log"
	"net/http"
	"path/filepath"
	"strings"

	"github.com/jrb/cuda-learning/webserver/internal/application"
	"github.com/jrb/cuda-learning/webserver/internal/infrastructure/processor"
	httpHandler "github.com/jrb/cuda-learning/webserver/internal/interfaces/http"
)

var (
	devMode     = flag.Bool("dev", false, "Enable development mode (hot reload templates and static files)")
	webRootPath = flag.String("webroot", "webserver/web", "Path to web assets (templates and static files)")
)

func main() {
	flag.Parse()

	if *devMode {
		log.Println("🔧 Development mode enabled - templates and static files will be read from disk")
		log.Printf("📁 Web root path: %s", *webRootPath)
	}

	// Dependency Injection - Clean Architecture
	// Infrastructure layer
	cppConnector := processor.NewCppConnector()

	// Application layer
	processImageUseCase := application.NewProcessImageUseCase(cppConnector)

	// Interface layer (HTTP)
	handler := httpHandler.NewHandler(processImageUseCase, *devMode, *webRootPath)

	// Static file serving
	http.HandleFunc("/static/", func(w http.ResponseWriter, r *http.Request) {
		// Strip /static/ prefix and serve from webroot/static/
		filePath := r.URL.Path[len("/static/"):]
		fullPath := filepath.Join(*webRootPath, "static", filePath)
		
		// Set proper content type
		if strings.HasSuffix(filePath, ".css") {
			w.Header().Set("Content-Type", "text/css")
		} else if strings.HasSuffix(filePath, ".js") {
			w.Header().Set("Content-Type", "application/javascript")
		}
		
		// In dev mode, disable caching
		if *devMode {
			w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
			w.Header().Set("Pragma", "no-cache")
			w.Header().Set("Expires", "0")
		}
		
		http.ServeFile(w, r, fullPath)
	})

	// Routes
	http.HandleFunc("/", handler.HandleIndex)
	http.HandleFunc("/ws", handler.HandleWebSocket)
	http.HandleFunc("/process", handler.HandleProcessImage)

	// Start server
	port := ":8080"
	log.Printf("🚀 Starting CUDA Image Processor Web Server on %s\n", port)
	log.Printf("📷 Open http://localhost%s in your browser\n", port)
	
	if err := http.ListenAndServe(port, nil); err != nil {
		log.Fatal(err)
	}
}

