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
	filterParam := r.URL.Query().Get("filter")
	if filterParam == "" {
		filterParam = "none"
	}
	
	acceleratorParam := r.URL.Query().Get("accelerator")
	if acceleratorParam == "" {
		acceleratorParam = "gpu"
	}
	
	grayscaleParam := r.URL.Query().Get("grayscale_type")
	if grayscaleParam == "" {
		grayscaleParam = "bt601"
	}
	
	var filters []domain.FilterType
	switch filterParam {
	case "none":
		filters = []domain.FilterType{domain.FilterNone}
	case "grayscale":
		filters = []domain.FilterType{domain.FilterGrayscale}
	default:
		filters = []domain.FilterType{domain.FilterNone}
	}
	
	var accelerator domain.AcceleratorType
	switch acceleratorParam {
	case "gpu":
		accelerator = domain.AcceleratorGPU
	case "cpu":
		accelerator = domain.AcceleratorCPU
	default:
		accelerator = domain.AcceleratorGPU
	}
	
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

	lenaPath := filepath.Join("data", "lena.png")
	imageFile, err := os.Open(lenaPath)
	if err != nil {
		log.Printf("open failed: %v", err)
		http.Error(w, "load failed", http.StatusInternalServerError)
		return
	}
	defer imageFile.Close()

	img, _, err := image.Decode(imageFile)
	if err != nil {
		log.Printf("decode failed: %v", err)
		http.Error(w, "decode failed", http.StatusInternalServerError)
		return
	}

	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	rgba := image.NewRGBA(bounds)
	
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			rgba.Set(x, y, img.At(x, y))
		}
	}

	domainImg := &domain.Image{
		Data:   rgba.Pix,
		Width:  width,
		Height: height,
		Format: "png",
	}

	processedImg, err := h.useCase.Execute(domainImg, filters, accelerator, grayscaleType)
	if err != nil {
		log.Printf("process failed: %v", err)
		http.Error(w, "process failed", http.StatusInternalServerError)
		return
	}

	var buf bytes.Buffer
	hasGrayscale := false
	for _, f := range filters {
		if f == domain.FilterGrayscale {
			hasGrayscale = true
			break
		}
	}
	
	if !hasGrayscale {
		resultImg := image.NewRGBA(image.Rect(0, 0, processedImg.Width, processedImg.Height))
		resultImg.Pix = processedImg.Data
		if err := png.Encode(&buf, resultImg); err != nil {
			log.Printf("encode failed: %v", err)
			http.Error(w, "encode failed", http.StatusInternalServerError)
			return
		}
	} else {
		grayImg := image.NewGray(image.Rect(0, 0, processedImg.Width, processedImg.Height))
		grayImg.Pix = processedImg.Data
		if err := png.Encode(&buf, grayImg); err != nil {
			log.Printf("encode failed: %v", err)
			http.Error(w, "encode failed", http.StatusInternalServerError)
			return
		}
	}

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

	tmpl := h.tmpl
	if h.devMode {
		var err error
		templatePath := filepath.Join(h.webRootPath, "templates", "index.html")
		tmpl, err = template.ParseFiles(templatePath)
		if err != nil {
			log.Printf("template parse failed: %v", err)
			http.Error(w, "template error", http.StatusInternalServerError)
			return
		}
	}

	if err := tmpl.Execute(w, data); err != nil {
		log.Printf("template exec failed: %v", err)
		http.Error(w, "render failed", http.StatusInternalServerError)
	}
}

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
		log.Printf("ws upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	for {
		_, message, err := conn.ReadMessage()
		if err != nil {
			break
		}

		var frameMsg FrameMessage
		if err := json.Unmarshal(message, &frameMsg); err != nil {
			continue
		}

		result := h.processFrame(&frameMsg)

		responseBytes, err := json.Marshal(result)
		if err != nil {
			continue
		}

		if err := conn.WriteMessage(1, responseBytes); err != nil {
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

	imageData, err := base64.StdEncoding.DecodeString(frameMsg.Image.Data)
	if err != nil {
		result.Error = "decode failed"
		return result
	}

	img, err := png.Decode(bytes.NewReader(imageData))
	if err != nil {
		img, err = jpeg.Decode(bytes.NewReader(imageData))
		if err != nil {
			result.Error = "image decode failed"
			return result
		}
	}

	bounds := img.Bounds()
	rgba := image.NewRGBA(bounds)
	for y := 0; y < bounds.Dy(); y++ {
		for x := 0; x < bounds.Dx(); x++ {
			rgba.Set(x, y, img.At(x, y))
		}
	}

	domainImg := &domain.Image{
		Data:   rgba.Pix,
		Width:  bounds.Dx(),
		Height: bounds.Dy(),
		Format: "png",
	}

	var filters []domain.FilterType
	for _, filterStr := range frameMsg.Filters {
		switch filterStr {
		case "none":
			filters = append(filters, domain.FilterNone)
		case "grayscale":
			filters = append(filters, domain.FilterGrayscale)
		}
	}
	
	if len(filters) == 0 {
		filters = []domain.FilterType{domain.FilterNone}
	}
	
	var accelerator domain.AcceleratorType
	switch frameMsg.Accelerator {
	case "cpu":
		accelerator = domain.AcceleratorCPU
	case "gpu":
		accelerator = domain.AcceleratorGPU
	default:
		accelerator = domain.AcceleratorGPU
	}
	
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

	h.frameCounter++
	processedImg, err := h.useCase.Execute(domainImg, filters, accelerator, grayscaleType)
	if err != nil {
		result.Error = "processing failed"
		return result
	}

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
			result.Error = "encode failed"
			return result
		}
	} else {
		grayImg := image.NewGray(image.Rect(0, 0, processedImg.Width, processedImg.Height))
		grayImg.Pix = processedImg.Data
		if err := png.Encode(&buf, grayImg); err != nil {
			result.Error = "encode failed"
			return result
		}
	}

	result.Success = true
	result.Image.Data = base64.StdEncoding.EncodeToString(buf.Bytes())
	result.Image.Width = processedImg.Width
	result.Image.Height = processedImg.Height

	elapsed := time.Since(startTime)
	if h.frameCounter%30 == 0 {
		log.Printf("frame %v (%dx%d %v %s %db)", 
			elapsed, processedImg.Width, processedImg.Height, frameMsg.Filters, frameMsg.Accelerator, len(buf.Bytes()))
	}

	return result
}

func (h *Handler) HandleProcessImage(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	img := &domain.Image{
		Data:   []byte{},
		Width:  512,
		Height: 512,
		Format: "png",
	}

	_, err := h.useCase.Execute(img, []domain.FilterType{domain.FilterGrayscale}, domain.AcceleratorGPU, domain.GrayscaleBT601)
	if err != nil {
		http.Error(w, "process failed", http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
}

