package http

import (
	"encoding/base64"
	"html/template"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/gorilla/websocket"
	"github.com/jrb/cuda-learning/webserver/internal/application"
	"github.com/jrb/cuda-learning/webserver/internal/domain"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

type Handler struct {
	useCase *application.ProcessImageUseCase
	tmpl    *template.Template
}

func NewHandler(useCase *application.ProcessImageUseCase) *Handler {
	tmpl := template.Must(template.ParseFiles("webserver/web/templates/index.html"))
	return &Handler{
		useCase: useCase,
		tmpl:    tmpl,
	}
}

func (h *Handler) HandleIndex(w http.ResponseWriter, r *http.Request) {
	// Load lena.png as base64 for initial display
	lenaPath := filepath.Join("data", "lena.png")
	imageData, err := os.ReadFile(lenaPath)
	if err != nil {
		log.Printf("Error reading lena.png: %v", err)
		http.Error(w, "Error loading image", http.StatusInternalServerError)
		return
	}

	base64Image := base64.StdEncoding.EncodeToString(imageData)

	data := struct {
		ImageData string
	}{
		ImageData: base64Image,
	}

	if err := h.tmpl.Execute(w, data); err != nil {
		log.Printf("Error executing template: %v", err)
		http.Error(w, "Error rendering page", http.StatusInternalServerError)
	}
}

func (h *Handler) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()

	log.Println("WebSocket connection established")

	for {
		messageType, message, err := conn.ReadMessage()
		if err != nil {
			log.Printf("WebSocket read error: %v", err)
			break
		}

		log.Printf("Received message: %s", message)

		// Echo back for now (future: process image and send back)
		if err := conn.WriteMessage(messageType, message); err != nil {
			log.Printf("WebSocket write error: %v", err)
			break
		}
	}
}

func (h *Handler) HandleProcessImage(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Future: Parse multipart form data for image upload
	// For now, this is a stub
	img := &domain.Image{
		Data:   []byte{},
		Width:  512,
		Height: 512,
		Format: "png",
	}

	processedImg, err := h.useCase.Execute(img, domain.FilterGrayscale)
	if err != nil {
		log.Printf("Error processing image: %v", err)
		http.Error(w, "Error processing image", http.StatusInternalServerError)
		return
	}

	log.Printf("Image processed: %dx%d", processedImg.Width, processedImg.Height)
	w.WriteHeader(http.StatusOK)
}

