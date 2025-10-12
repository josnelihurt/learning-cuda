package static_http

import (
	"bytes"
	"context"
	"image"
	"image/jpeg"
	"image/png"
	"log"
	"net/http"
	"time"

	"connectrpc.com/connect"
	"github.com/gorilla/websocket"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/webserver/internal/config"
	"github.com/jrb/cuda-learning/webserver/internal/interfaces/connectrpc"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

type WebSocketHandler struct {
	rpcHandler   *connectrpc.ImageProcessorHandler
	config       *config.Config
	frameCounter int
}

func NewWebSocketHandler(rpcHandler *connectrpc.ImageProcessorHandler, cfg *config.Config) *WebSocketHandler {
	return &WebSocketHandler{
		rpcHandler:   rpcHandler,
		config:       cfg,
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

	transportFormat := h.config.Stream.TransportFormat
	log.Printf("WebSocket connected, transport format: %s", transportFormat)

	for {
		messageType, message, err := conn.ReadMessage()
		if err != nil {
			break
		}

		var frameMsg pb.WebSocketFrameRequest
		if transportFormat == "binary" {
			err = proto.Unmarshal(message, &frameMsg)
		} else {
			err = protojson.Unmarshal(message, &frameMsg)
		}

		if err != nil {
			log.Printf("failed to unmarshal message: %v", err)
			continue
		}

		result := h.processFrame(&frameMsg)

		var responseBytes []byte
		if transportFormat == "binary" {
			responseBytes, err = proto.Marshal(result)
		} else {
			responseBytes, err = protojson.Marshal(result)
		}

		if err != nil {
			log.Printf("failed to marshal response: %v", err)
			continue
		}

		if err := conn.WriteMessage(messageType, responseBytes); err != nil {
			break
		}
	}
}

func (h *WebSocketHandler) processFrame(frameMsg *pb.WebSocketFrameRequest) *pb.WebSocketFrameResponse {
	startTime := time.Now()
	result := &pb.WebSocketFrameResponse{
		Type:    "frame_result",
		Success: false,
	}

	if frameMsg.Request == nil {
		result.Error = "missing request"
		return result
	}

	req := frameMsg.Request

	if len(req.ImageData) == 0 {
		result.Error = "missing image data"
		return result
	}

	img, err := png.Decode(bytes.NewReader(req.ImageData))
	if err != nil {
		img, err = jpeg.Decode(bytes.NewReader(req.ImageData))
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

	req.ImageData = rgba.Pix
	req.Width = int32(bounds.Dx())
	req.Height = int32(bounds.Dy())
	req.Channels = 4

	h.frameCounter++
	resp, err := h.rpcHandler.ProcessImage(context.Background(), connect.NewRequest(req))
	if err != nil {
		result.Error = "processing failed: " + err.Error()
		return result
	}

	hasGrayscale := false
	for _, f := range req.Filters {
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
	result.Response = &pb.ProcessImageResponse{
		Code:      0,
		Message:   "success",
		ImageData: buf.Bytes(),
		Width:     resp.Msg.Width,
		Height:    resp.Msg.Height,
		Channels:  resp.Msg.Channels,
	}

	elapsed := time.Since(startTime)
	if h.frameCounter%30 == 0 {
		log.Printf("frame %v (%dx%d %v %v)", 
			elapsed, resp.Msg.Width, resp.Msg.Height, req.Filters, req.Accelerator)
	}

	return result
}

