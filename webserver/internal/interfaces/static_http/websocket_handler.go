package static_http

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/jpeg"
	"image/png"
	"log"
	"net/http"
	"strings"
	"time"

	"connectrpc.com/connect"
	"github.com/gorilla/websocket"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/webserver/internal/interfaces/connectrpc"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

type FrameMessage struct {
	Type          string   `json:"type"`
	Filters       []string `json:"filters"`
	Accelerator   string   `json:"accelerator"`
	GrayscaleType string   `json:"grayscale_type"`
	Image         struct {
		Data     string `json:"data"`
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
		Data   string `json:"data"`
		Width  int    `json:"width"`
		Height int    `json:"height"`
	} `json:"image"`
}

type WebSocketHandler struct {
	rpcHandler   *connectrpc.ImageProcessorHandler
	frameCounter int
}

func NewWebSocketHandler(rpcHandler *connectrpc.ImageProcessorHandler) *WebSocketHandler {
	return &WebSocketHandler{
		rpcHandler:   rpcHandler,
		frameCounter: 0,
	}
}

func (h *WebSocketHandler) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
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

func (h *WebSocketHandler) processFrame(frameMsg *FrameMessage) *FrameResultMessage {
	startTime := time.Now()
	result := &FrameResultMessage{
		Type:    "frame_result",
		Success: false,
	}

	imageDataB64 := strings.TrimPrefix(frameMsg.Image.Data, "data:image/png;base64,")
	pngData, err := base64.StdEncoding.DecodeString(imageDataB64)
	if err != nil {
		result.Error = "decode failed"
		return result
	}

	img, err := png.Decode(bytes.NewReader(pngData))
	if err != nil {
		img, err = jpeg.Decode(bytes.NewReader(pngData))
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

	var protoFilters []pb.FilterType
	for _, filterStr := range frameMsg.Filters {
		switch filterStr {
		case "none":
			protoFilters = append(protoFilters, pb.FilterType_FILTER_TYPE_NONE)
		case "grayscale":
			protoFilters = append(protoFilters, pb.FilterType_FILTER_TYPE_GRAYSCALE)
		}
	}
	
	var protoAccelerator pb.AcceleratorType
	switch frameMsg.Accelerator {
	case "cpu":
		protoAccelerator = pb.AcceleratorType_ACCELERATOR_TYPE_CPU
	case "gpu":
		protoAccelerator = pb.AcceleratorType_ACCELERATOR_TYPE_GPU
	default:
		protoAccelerator = pb.AcceleratorType_ACCELERATOR_TYPE_GPU
	}
	
	var protoGrayscaleType pb.GrayscaleType
	switch frameMsg.GrayscaleType {
	case "bt601":
		protoGrayscaleType = pb.GrayscaleType_GRAYSCALE_TYPE_BT601
	case "bt709":
		protoGrayscaleType = pb.GrayscaleType_GRAYSCALE_TYPE_BT709
	case "average":
		protoGrayscaleType = pb.GrayscaleType_GRAYSCALE_TYPE_AVERAGE
	case "lightness":
		protoGrayscaleType = pb.GrayscaleType_GRAYSCALE_TYPE_LIGHTNESS
	case "luminosity":
		protoGrayscaleType = pb.GrayscaleType_GRAYSCALE_TYPE_LUMINOSITY
	default:
		protoGrayscaleType = pb.GrayscaleType_GRAYSCALE_TYPE_BT601
	}

	req := &pb.ProcessImageRequest{
		ImageData:     rgba.Pix,
		Width:         int32(bounds.Dx()),
		Height:        int32(bounds.Dy()),
		Channels:      4,
		Filters:       protoFilters,
		Accelerator:   protoAccelerator,
		GrayscaleType: protoGrayscaleType,
	}

	h.frameCounter++
	resp, err := h.rpcHandler.ProcessImage(context.Background(), connect.NewRequest(req))
	if err != nil {
		result.Error = "processing failed"
		return result
	}

	hasGrayscale := false
	for _, f := range protoFilters {
		if f == pb.FilterType_FILTER_TYPE_GRAYSCALE {
			hasGrayscale = true
			break
		}
	}

	var buf bytes.Buffer
	if !hasGrayscale {
		resultImg := image.NewRGBA(image.Rect(0, 0, int(resp.Msg.Width), int(resp.Msg.Height)))
		resultImg.Pix = resp.Msg.ImageData
		if err := png.Encode(&buf, resultImg); err != nil {
			result.Error = "encode failed"
			return result
		}
	} else {
		grayImg := image.NewGray(image.Rect(0, 0, int(resp.Msg.Width), int(resp.Msg.Height)))
		grayImg.Pix = resp.Msg.ImageData
		if err := png.Encode(&buf, grayImg); err != nil {
			result.Error = "encode failed"
			return result
		}
	}

	result.Success = true
	result.Image.Data = base64.StdEncoding.EncodeToString(buf.Bytes())
	result.Image.Width = int(resp.Msg.Width)
	result.Image.Height = int(resp.Msg.Height)

	elapsed := time.Since(startTime)
	if h.frameCounter%30 == 0 {
		log.Printf("frame %v (%dx%d %v %s)", 
			elapsed, resp.Msg.Width, resp.Msg.Height, frameMsg.Filters, frameMsg.Accelerator)
	}

	return result
}

