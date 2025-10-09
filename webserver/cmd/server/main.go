package main

import (
	"log"
	"net/http"

	"github.com/jrb/cuda-learning/webserver/internal/application"
	"github.com/jrb/cuda-learning/webserver/internal/infrastructure/processor"
	httpHandler "github.com/jrb/cuda-learning/webserver/internal/interfaces/http"
)

func main() {
	// Dependency Injection - Clean Architecture
	// Infrastructure layer
	cppConnector := processor.NewCppConnector()

	// Application layer
	processImageUseCase := application.NewProcessImageUseCase(cppConnector)

	// Interface layer (HTTP)
	handler := httpHandler.NewHandler(processImageUseCase)

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

