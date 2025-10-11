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
	"time"

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
	useCase      *application.ProcessImageUseCase
	tmpl         *template.Template
	devMode      bool
	webRootPath  string
	frameCounter int
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
	// Get filters from query parameter (comma-separated), default to "none"
	filterParam := r.URL.Query().Get("filter")
	if filterParam == "" {
		filterParam = "none"
	}
	
	// Get accelerator type, default to "gpu"
	acceleratorParam := r.URL.Query().Get("accelerator")
	if acceleratorParam == "" {
		acceleratorParam = "gpu"
	}
	
	// Get grayscale type, default to "bt601"
	grayscaleParam := r.URL.Query().Get("grayscale_type")
	if grayscaleParam == "" {
		grayscaleParam = "bt601"
	}
	
	// Parse filters (for now, single filter support from query param)
	var filters []domain.FilterType
	switch filterParam {
	case "none":
		filters = []domain.FilterType{domain.FilterNone}
	case "grayscale":
		filters = []domain.FilterType{domain.FilterGrayscale}
	default:
		filters = []domain.FilterType{domain.FilterNone}
	}
	
	// Parse accelerator type
	var accelerator domain.AcceleratorType
	switch acceleratorParam {
	case "gpu":
		accelerator = domain.AcceleratorGPU
	case "cpu":
		accelerator = domain.AcceleratorCPU
	default:
		accelerator = domain.AcceleratorGPU
	}
	
	// Parse grayscale type
	var grayscaleType domain.GrayscaleType
	switch grayscaleParam {
	case "bt601":
		grayscaleType = domain.GrayscaleBT601
	case "bt709":
		grayscaleType = domain.GrayscaleBT709
	case "average":
		grayscaleType = domain.GrayscaleAverage
	case "lightness":
		grayscaleType = domain.GrayscaleLightness
	case "luminosity":
		grayscaleType = domain.GrayscaleLuminosity
	default:
		grayscaleType = domain.GrayscaleBT601
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

	// Process with selected filters
	processedImg, err := h.useCase.Execute(domainImg, filters, accelerator, grayscaleType)
	if err != nil {
		log.Printf("Error processing image: %v", err)
		http.Error(w, "Error processing image", http.StatusInternalServerError)
		return
	}

	// Encode result to PNG based on filter type
	var buf bytes.Buffer
	hasGrayscale := false
	for _, f := range filters {
		if f == domain.FilterGrayscale {
			hasGrayscale = true
			break
		}
	}
	
	if !hasGrayscale {
		// Original image in RGBA format
		resultImg := image.NewRGBA(image.Rect(0, 0, processedImg.Width, processedImg.Height))
		resultImg.Pix = processedImg.Data
		if err := png.Encode(&buf, resultImg); err != nil {
			log.Printf("Error encoding PNG: %v", err)
			http.Error(w, "Error encoding image", http.StatusInternalServerError)
			return
		}
	} else {
		// Grayscale image
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
		ImageData         string
		SelectedFilter    string
		SelectedAccel     string
		SelectedGrayscale string
	}{
		ImageData:         base64Image,
		SelectedFilter:    filterParam,
		SelectedAccel:     acceleratorParam,
		SelectedGrayscale: grayscaleParam,
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
	Type          string   `json:"type"`
	Filters       []string `json:"filters"`       // Array of filter names
	Accelerator   string   `json:"accelerator"`   // "gpu" or "cpu"
	GrayscaleType string   `json:"grayscale_type"` // grayscale algorithm
	Image         struct {
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
	startTime := time.Now()
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

	// Map filters
	var filters []domain.FilterType
	for _, filterStr := range frameMsg.Filters {
		switch filterStr {
		case "none":
			filters = append(filters, domain.FilterNone)
		case "grayscale":
			filters = append(filters, domain.FilterGrayscale)
		}
	}
	
	// Default to none if empty
	if len(filters) == 0 {
		filters = []domain.FilterType{domain.FilterNone}
	}
	
	// Map accelerator type
	var accelerator domain.AcceleratorType
	switch frameMsg.Accelerator {
	case "cpu":
		accelerator = domain.AcceleratorCPU
	case "gpu":
		accelerator = domain.AcceleratorGPU
	default:
		accelerator = domain.AcceleratorGPU
	}
	
	// Map grayscale type
	var grayscaleType domain.GrayscaleType
	switch frameMsg.GrayscaleType {
	case "bt601":
		grayscaleType = domain.GrayscaleBT601
	case "bt709":
		grayscaleType = domain.GrayscaleBT709
	case "average":
		grayscaleType = domain.GrayscaleAverage
	case "lightness":
		grayscaleType = domain.GrayscaleLightness
	case "luminosity":
		grayscaleType = domain.GrayscaleLuminosity
	default:
		grayscaleType = domain.GrayscaleBT601
	}

	// Process with CUDA/CPU
	h.frameCounter++
	processedImg, err := h.useCase.Execute(domainImg, filters, accelerator, grayscaleType)
	if err != nil {
		result.Error = "Processing failed"
		log.Printf("Processing error: %v", err)
		return result
	}

	// Encode result based on filters
	hasGrayscale := false
	for _, f := range filters {
		if f == domain.FilterGrayscale {
			hasGrayscale = true
			break
		}
	}
	
	var buf bytes.Buffer
	if !hasGrayscale {
		resultImg := image.NewRGBA(image.Rect(0, 0, processedImg.Width, processedImg.Height))
		resultImg.Pix = processedImg.Data
		if err := png.Encode(&buf, resultImg); err != nil {
			result.Error = "Failed to encode result"
			return result
		}
	} else {
		grayImg := image.NewGray(image.Rect(0, 0, processedImg.Width, processedImg.Height))
		grayImg.Pix = processedImg.Data
		if err := png.Encode(&buf, grayImg); err != nil {
			result.Error = "Failed to encode result"
			return result
		}
	}

	// Success
	result.Success = true
	result.Image.Data = base64.StdEncoding.EncodeToString(buf.Bytes())
	result.Image.Width = processedImg.Width
	result.Image.Height = processedImg.Height

	// Log performance every 30 frames
	elapsed := time.Since(startTime)
	if h.frameCounter%30 == 0 {
		log.Printf("Frame processing: %v (%dx%d, filters: %v, accelerator: %s, size: %d bytes)", 
			elapsed, processedImg.Width, processedImg.Height, frameMsg.Filters, frameMsg.Accelerator, len(buf.Bytes()))
	}

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

	processedImg, err := h.useCase.Execute(img, []domain.FilterType{domain.FilterGrayscale}, domain.AcceleratorGPU, domain.GrayscaleBT601)
	if err != nil {
		log.Printf("Error processing image: %v", err)
		http.Error(w, "Error processing image", http.StatusInternalServerError)
		return
	}

	log.Printf("Image processed: %dx%d", processedImg.Width, processedImg.Height)
	w.WriteHeader(http.StatusOK)
}

