package http

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"html/template"
	"image"
	"image/jpeg"
	"image/png"
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
	useCase     *application.ProcessImageUseCase
	tmpl        *template.Template
	devMode     bool
	webRootPath string
}

func NewHandler(useCase *application.ProcessImageUseCase, devMode bool, webRootPath string) *Handler {
	var tmpl *template.Template
	if !devMode {
		// Production: Load template once at startup
		templatePath := filepath.Join(webRootPath, "templates", "index.html")
		tmpl = template.Must(template.ParseFiles(templatePath))
	}
	return &Handler{
		useCase:     useCase,
		tmpl:        tmpl,
		devMode:     devMode,
		webRootPath: webRootPath,
	}
}

func (h *Handler) HandleIndex(w http.ResponseWriter, r *http.Request) {
	// Get filter from query parameter, default to "none"
	filterParam := r.URL.Query().Get("filter")
	if filterParam == "" {
		filterParam = "none"
	}
	
	var filter domain.FilterType
	switch filterParam {
	case "none":
		filter = domain.FilterNone
	case "grayscale":
		filter = domain.FilterGrayscale
	default:
		filter = domain.FilterNone
	}

	// Load lena.png
	lenaPath := filepath.Join("data", "lena.png")
	imageFile, err := os.Open(lenaPath)
	if err != nil {
		log.Printf("Error reading lena.png: %v", err)
		http.Error(w, "Error loading image", http.StatusInternalServerError)
		return
	}
	defer imageFile.Close()

	// Decode PNG to get raw pixels
	img, _, err := image.Decode(imageFile)
	if err != nil {
		log.Printf("Error decoding PNG: %v", err)
		http.Error(w, "Error decoding image", http.StatusInternalServerError)
		return
	}

	// Convert to RGBA format
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	rgba := image.NewRGBA(bounds)
	
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			rgba.Set(x, y, img.At(x, y))
		}
	}

	// Create domain image with raw pixel data
	domainImg := &domain.Image{
		Data:   rgba.Pix,
		Width:  width,
		Height: height,
		Format: "png",
	}

	log.Printf("Processing image with filter '%s': %dx%d, %d bytes", filter, width, height, len(domainImg.Data))

	// Process with selected filter
	processedImg, err := h.useCase.Execute(domainImg, filter)
	if err != nil {
		log.Printf("Error processing image: %v", err)
		http.Error(w, "Error processing image", http.StatusInternalServerError)
		return
	}

	// Encode result to PNG based on filter type
	var buf bytes.Buffer
	if filter == domain.FilterNone {
		// Original image in RGBA format
		log.Printf("Returning original image: %dx%d, RGBA", processedImg.Width, processedImg.Height)
		resultImg := image.NewRGBA(image.Rect(0, 0, processedImg.Width, processedImg.Height))
		resultImg.Pix = processedImg.Data
		if err := png.Encode(&buf, resultImg); err != nil {
			log.Printf("Error encoding PNG: %v", err)
			http.Error(w, "Error encoding image", http.StatusInternalServerError)
			return
		}
	} else {
		// Grayscale image
		log.Printf("Processed image: %dx%d, grayscale, %d bytes", 
			processedImg.Width, processedImg.Height, len(processedImg.Data))
		grayImg := image.NewGray(image.Rect(0, 0, processedImg.Width, processedImg.Height))
		grayImg.Pix = processedImg.Data
		if err := png.Encode(&buf, grayImg); err != nil {
			log.Printf("Error encoding PNG: %v", err)
			http.Error(w, "Error encoding image", http.StatusInternalServerError)
			return
		}
	}

	// Convert to base64
	base64Image := base64.StdEncoding.EncodeToString(buf.Bytes())

	data := struct {
		ImageData      string
		SelectedFilter string
	}{
		ImageData:      base64Image,
		SelectedFilter: filterParam,
	}

	// In dev mode, reload template on each request
	tmpl := h.tmpl
	if h.devMode {
		var err error
		templatePath := filepath.Join(h.webRootPath, "templates", "index.html")
		tmpl, err = template.ParseFiles(templatePath)
		if err != nil {
			log.Printf("Error parsing template: %v", err)
			http.Error(w, "Error loading template", http.StatusInternalServerError)
			return
		}
	}

	if err := tmpl.Execute(w, data); err != nil {
		log.Printf("Error executing template: %v", err)
		http.Error(w, "Error rendering page", http.StatusInternalServerError)
	}
}

// WebSocket message types
type FrameMessage struct {
	Type   string `json:"type"`
	Filter string `json:"filter"`
	Image  struct {
		Data     string `json:"data"` // base64 encoded PNG
		Width    int    `json:"width"`
		Height   int    `json:"height"`
		Channels int    `json:"channels"`
	} `json:"image"`
}

type FrameResultMessage struct {
	Type    string `json:"type"`
	Success bool   `json:"success"`
	Error   string `json:"error,omitempty"`
	Image   struct {
		Data   string `json:"data"` // base64 encoded PNG
		Width  int    `json:"width"`
		Height int    `json:"height"`
	} `json:"image"`
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
		_, message, err := conn.ReadMessage()
		if err != nil {
			log.Printf("WebSocket read error: %v", err)
			break
		}

		// Parse incoming frame message
		var frameMsg FrameMessage
		if err := json.Unmarshal(message, &frameMsg); err != nil {
			log.Printf("Error parsing frame message: %v", err)
			continue
		}

		// Process frame
		result := h.processFrame(&frameMsg)

		// Send result back
		responseBytes, err := json.Marshal(result)
		if err != nil {
			log.Printf("Error marshaling response: %v", err)
			continue
		}

		if err := conn.WriteMessage(1, responseBytes); err != nil {
			log.Printf("WebSocket write error: %v", err)
			break
		}
	}
}

func (h *Handler) processFrame(frameMsg *FrameMessage) *FrameResultMessage {
	result := &FrameResultMessage{
		Type:    "frame_result",
		Success: false,
	}

	// Decode base64 image data (PNG or JPEG)
	imageData, err := base64.StdEncoding.DecodeString(frameMsg.Image.Data)
	if err != nil {
		result.Error = "Failed to decode base64 data"
		log.Printf("Base64 decode error: %v", err)
		return result
	}

	// Decode image (try PNG first, then JPEG)
	img, err := png.Decode(bytes.NewReader(imageData))
	if err != nil {
		// Try JPEG
		img, err = jpeg.Decode(bytes.NewReader(imageData))
		if err != nil {
			result.Error = "Failed to decode image"
			log.Printf("Image decode error: %v", err)
			return result
		}
	}

	// Convert to RGBA
	bounds := img.Bounds()
	rgba := image.NewRGBA(bounds)
	for y := 0; y < bounds.Dy(); y++ {
		for x := 0; x < bounds.Dx(); x++ {
			rgba.Set(x, y, img.At(x, y))
		}
	}

	// Create domain image
	domainImg := &domain.Image{
		Data:   rgba.Pix,
		Width:  bounds.Dx(),
		Height: bounds.Dy(),
		Format: "png",
	}

	// Map filter
	var filter domain.FilterType
	switch frameMsg.Filter {
	case "none":
		filter = domain.FilterNone
	case "grayscale":
		filter = domain.FilterGrayscale
	default:
		filter = domain.FilterNone
	}

	// Process with CUDA
	processedImg, err := h.useCase.Execute(domainImg, filter)
	if err != nil {
		result.Error = "CUDA processing failed"
		log.Printf("Processing error: %v", err)
		return result
	}

	log.Printf("Frame processed: %dx%d, filter: %s, data len: %d", 
		processedImg.Width, processedImg.Height, filter, len(processedImg.Data))

	// Encode result based on filter
	var buf bytes.Buffer
	if filter == domain.FilterNone {
		resultImg := image.NewRGBA(image.Rect(0, 0, processedImg.Width, processedImg.Height))
		resultImg.Pix = processedImg.Data
		if err := png.Encode(&buf, resultImg); err != nil {
			result.Error = "Failed to encode result"
			log.Printf("PNG encode error (RGBA): %v", err)
			return result
		}
	} else {
		grayImg := image.NewGray(image.Rect(0, 0, processedImg.Width, processedImg.Height))
		grayImg.Pix = processedImg.Data
		
		// Debug: Check if data is all zeros
		nonZero := 0
		checkLen := 100
		if len(processedImg.Data) < checkLen {
			checkLen = len(processedImg.Data)
		}
		for i := 0; i < checkLen; i++ {
			if processedImg.Data[i] != 0 {
				nonZero++
			}
		}
		log.Printf("Grayscale data check - first %d bytes, non-zero: %d", checkLen, nonZero)
		
		if err := png.Encode(&buf, grayImg); err != nil {
			result.Error = "Failed to encode result"
			log.Printf("PNG encode error (Gray): %v", err)
			return result
		}
	}
	
	log.Printf("Encoded PNG size: %d bytes", buf.Len())

	// Success
	result.Success = true
	result.Image.Data = base64.StdEncoding.EncodeToString(buf.Bytes())
	result.Image.Width = processedImg.Width
	result.Image.Height = processedImg.Height

	return result
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

