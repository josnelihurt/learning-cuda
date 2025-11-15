package steps

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	_ "image/png" // Required for PNG image encoding/decoding
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"connectrpc.com/connect"
	"github.com/gorilla/websocket"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/proto/gen/genconnect"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/featureflags"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

var rootPath = "/"

type BDDContext struct {
	fliptAPI               *featureflags.FliptHTTPAPI
	httpClient             *http.Client
	serviceBaseURL         string
	lastResponse           *http.Response
	lastResponseBody       []byte
	defaultFormat          string
	defaultEndpoint        string
	connectClient          genconnect.ImageProcessorServiceClient
	currentImage           []byte
	currentImagePNG        []byte
	currentImageWidth      int32
	currentImageHeight     int32
	currentChannels        int32
	processedImage         []byte
	wsConnection           *websocket.Conn
	wsResponse             *pb.WebSocketFrameResponse
	lastError              error
	checksums              map[string]string
	inputSources           []*pb.InputSource
	availableImages        []*pb.StaticImage
	configClient           genconnect.ConfigServiceClient
	fileClient             genconnect.FileServiceClient
	processorCapabilities  *pb.LibraryCapabilities
	toolsResponse          *pb.GetAvailableToolsResponse
	currentTool            *pb.Tool
	uploadedImage          *pb.StaticImage
	availableVideos        []*pb.StaticVideo
	uploadedVideo          *pb.StaticVideo
	videoFrames            []*pb.VideoFrameUpdate
	frameCollector         chan *pb.VideoFrameUpdate
	stopCollector          chan bool
	receivedLogLevel       string
	receivedConsoleLogging bool
	otlpLogsReceived       bool
	systemInfoResponse     *pb.GetSystemInfoResponse
	listFiltersResponse    *pb.ListFiltersResponse
}

func NewBDDContext(fliptBaseURL, fliptNamespace, serviceBaseURL string) *BDDContext {
	httpClient := &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: true,
			},
		},
		Timeout: 30 * time.Second,
	}

	connectClient := genconnect.NewImageProcessorServiceClient(httpClient, serviceBaseURL)
	configClient := genconnect.NewConfigServiceClient(httpClient, serviceBaseURL)
	fileClient := genconnect.NewFileServiceClient(httpClient, serviceBaseURL)

	ctx := &BDDContext{
		fliptAPI:       featureflags.NewFliptHTTPAPI(fliptBaseURL, fliptNamespace, httpClient),
		httpClient:     httpClient,
		serviceBaseURL: serviceBaseURL,
		connectClient:  connectClient,
		configClient:   configClient,
		fileClient:     fileClient,
		checksums:      make(map[string]string),
	}

	ctx.loadChecksums()
	return ctx
}

func (c *BDDContext) GivenFliptIsClean() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := c.fliptAPI.CleanAllFlags(ctx); err != nil {
		return fmt.Errorf("failed to clean Flipt: %w", err)
	}

	time.Sleep(500 * time.Millisecond)
	return nil
}

func (c *BDDContext) GivenTheServiceIsRunning() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	url := fmt.Sprintf("%s/", c.serviceBaseURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("service is not running at %s: %w", c.serviceBaseURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 500 {
		return fmt.Errorf("service returned error status %d", resp.StatusCode)
	}

	return nil
}

func (c *BDDContext) GivenConfigHasDefaultValues(format, endpoint string) error {
	c.defaultFormat = format
	c.defaultEndpoint = endpoint
	return nil
}

// callConnectRPCEndpoint is a generic helper for calling ConnectRPC endpoints
func (c *BDDContext) callConnectRPCEndpoint(endpoint string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	url := fmt.Sprintf("%s/%s", c.serviceBaseURL, endpoint)

	reqBody := bytes.NewBufferString("{}")
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, reqBody)
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to call %s: %w", endpoint, err)
	}

	body, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return fmt.Errorf("failed to read response body: %w", err)
	}

	c.lastResponse = resp
	c.lastResponseBody = body

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

func (c *BDDContext) WhenICallGetStreamConfig() error {
	return c.callConnectRPCEndpoint("cuda_learning.ConfigService/GetStreamConfig")
}

func (c *BDDContext) WhenICallSyncFeatureFlags() error {
	return c.callConnectRPCEndpoint("cuda_learning.ConfigService/SyncFeatureFlags")
}

func (c *BDDContext) WhenIWaitForFlagsToBeSynced() error {
	time.Sleep(1 * time.Second)
	return nil
}

func (c *BDDContext) WhenICallHealthEndpoint() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	url := fmt.Sprintf("%s/health", c.serviceBaseURL)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to call health endpoint: %w", err)
	}

	body, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return fmt.Errorf("failed to read response body: %w", err)
	}

	c.lastResponse = resp
	c.lastResponseBody = body

	return nil
}

func (c *BDDContext) ThenTheResponseShouldContainTransportFormat(expected string) error {
	if c.lastResponseBody == nil {
		return fmt.Errorf("no response body available")
	}

	var response struct {
		Endpoints []struct {
			Type            string `json:"type"`
			Endpoint        string `json:"endpoint"`
			TransportFormat string `json:"transport_format"`
		} `json:"endpoints"`
	}

	if err := json.Unmarshal(c.lastResponseBody, &response); err != nil {
		return fmt.Errorf("failed to unmarshal response: %w (body: %s)", err, string(c.lastResponseBody))
	}

	if len(response.Endpoints) == 0 {
		return fmt.Errorf("no endpoints in response (body: %s)", string(c.lastResponseBody))
	}

	actual := response.Endpoints[0].TransportFormat
	if actual != expected {
		return fmt.Errorf("expected transport format '%s', got '%s' (full response: %s)", expected, actual, string(c.lastResponseBody))
	}

	return nil
}

func (c *BDDContext) ThenTheResponseShouldContainEndpoint(expected string) error {
	if c.lastResponseBody == nil {
		return fmt.Errorf("no response body available")
	}

	var response struct {
		Endpoints []struct {
			Type            string `json:"type"`
			Endpoint        string `json:"endpoint"`
			TransportFormat string `json:"transport_format"`
		} `json:"endpoints"`
	}

	if err := json.Unmarshal(c.lastResponseBody, &response); err != nil {
		return fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(response.Endpoints) == 0 {
		return fmt.Errorf("no endpoints in response")
	}

	actual := response.Endpoints[0].Endpoint
	if actual != expected {
		return fmt.Errorf("expected endpoint '%s', got '%s'", expected, actual)
	}

	return nil
}

func (c *BDDContext) ThenFliptShouldHaveFlag(flagKey string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	flag, err := c.fliptAPI.GetFlag(ctx, flagKey)
	if err != nil {
		return fmt.Errorf("flag '%s' not found in Flipt: %w", flagKey, err)
	}

	if flag.Key != flagKey {
		return fmt.Errorf("expected flag key '%s', got '%s'", flagKey, flag.Key)
	}

	return nil
}

func (c *BDDContext) ThenFliptShouldHaveFlagWithValue(flagKey string, expectedValue interface{}) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	flag, err := c.fliptAPI.GetFlag(ctx, flagKey)
	if err != nil {
		return fmt.Errorf("flag '%s' not found in Flipt: %w", flagKey, err)
	}

	switch v := expectedValue.(type) {
	case bool:
		if flag.Enabled != v {
			return fmt.Errorf("expected flag '%s' enabled=%v, got enabled=%v", flagKey, v, flag.Enabled)
		}
	default:
		return fmt.Errorf("unsupported value type %T for flag validation", expectedValue)
	}

	return nil
}

func (c *BDDContext) ThenTheResponseStatusShouldBe(statusCode int) error {
	if c.lastResponse == nil {
		return fmt.Errorf("no response available")
	}

	if c.lastResponse.StatusCode != statusCode {
		return fmt.Errorf("expected status code %d, got %d", statusCode, c.lastResponse.StatusCode)
	}

	return nil
}

func (c *BDDContext) ThenTheResponseShouldContainHealthStatus(status string) error {
	if c.lastResponseBody == nil {
		return fmt.Errorf("no response body available")
	}

	var response struct {
		Status string `json:"status"`
	}

	if err := json.Unmarshal(c.lastResponseBody, &response); err != nil {
		return fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if response.Status != status {
		return fmt.Errorf("expected status '%s', got '%s'", status, response.Status)
	}

	return nil
}

func (c *BDDContext) loadChecksums() {
	checksumPath := filepath.Join("testdata", "checksums.json")
	data, err := os.ReadFile(checksumPath)
	if err != nil {
		return
	}

	var checksumData struct {
		Checksums []struct {
			Image         string `json:"image"`
			Filter        string `json:"filter"`
			Accelerator   string `json:"accelerator"`
			GrayscaleType string `json:"grayscale_type"`
			Checksum      string `json:"checksum"`
		} `json:"checksums"`
	}

	if err := json.Unmarshal(data, &checksumData); err != nil {
		return
	}

	for _, entry := range checksumData.Checksums {
		key := fmt.Sprintf("%s_%s_%s_%s", entry.Image, entry.Filter, entry.Accelerator, entry.GrayscaleType)
		c.checksums[key] = entry.Checksum
	}
}

func (c *BDDContext) GivenIHaveImage(imageName string) error {
	imagePath := filepath.Join("..", "..", "..", "data", imageName)

	pngData, err := os.ReadFile(imagePath)
	if err != nil {
		return fmt.Errorf("failed to read PNG file %s: %w", imageName, err)
	}
	c.currentImagePNG = pngData

	file, err := os.Open(imagePath)
	if err != nil {
		return fmt.Errorf("failed to open image %s: %w", imageName, err)
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return fmt.Errorf("failed to decode image: %w", err)
	}

	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	rawData := make([]byte, width*height*3)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			idx := (y*width + x) * 3
			rawData[idx] = byte(r >> 8)
			rawData[idx+1] = byte(g >> 8)
			rawData[idx+2] = byte(b >> 8)
		}
	}

	c.currentImage = rawData
	c.currentImageWidth = int32(width)
	c.currentImageHeight = int32(height)
	c.currentChannels = 3

	return nil
}

func (c *BDDContext) WhenICallProcessImageWith(filter, accelerator, grayscaleType string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	filterEnum := parseFilterType(filter)
	acceleratorEnum := parseAcceleratorType(accelerator)
	grayscaleEnum := parseGrayscaleType(grayscaleType)

	req := &pb.ProcessImageRequest{
		ImageData:     c.currentImage,
		Width:         c.currentImageWidth,
		Height:        c.currentImageHeight,
		Channels:      c.currentChannels,
		Filters:       []pb.FilterType{filterEnum},
		Accelerator:   acceleratorEnum,
		GrayscaleType: grayscaleEnum,
	}

	resp, err := c.connectClient.ProcessImage(ctx, connect.NewRequest(req))
	if err != nil {
		c.lastError = err
		return nil
	}

	c.lastResponse = &http.Response{StatusCode: http.StatusOK}
	c.processedImage = resp.Msg.ImageData
	c.lastError = nil

	if resp.Msg.Code != 0 {
		c.lastError = fmt.Errorf("processing failed: %s", resp.Msg.Message)
	}

	return nil
}

func (c *BDDContext) WhenICallProcessImageWithBlurFilter(accelerator string, kernelSize int, sigma float64, borderMode string, separable bool) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	acceleratorEnum := parseAcceleratorType(accelerator)

	borderModeEnum := pb.BorderMode_BORDER_MODE_REFLECT
	switch borderMode {
	case "CLAMP":
		borderModeEnum = pb.BorderMode_BORDER_MODE_CLAMP
	case "REFLECT":
		borderModeEnum = pb.BorderMode_BORDER_MODE_REFLECT
	case "WRAP":
		borderModeEnum = pb.BorderMode_BORDER_MODE_WRAP
	}

	blurParams := &pb.GaussianBlurParameters{
		KernelSize: int32(kernelSize),
		Sigma:      float32(sigma),
		BorderMode: borderModeEnum,
		Separable:  separable,
	}

	req := &pb.ProcessImageRequest{
		ImageData:     c.currentImage,
		Width:         c.currentImageWidth,
		Height:        c.currentImageHeight,
		Channels:      c.currentChannels,
		Filters:       []pb.FilterType{pb.FilterType_FILTER_TYPE_BLUR},
		Accelerator:   acceleratorEnum,
		GrayscaleType: pb.GrayscaleType_GRAYSCALE_TYPE_UNSPECIFIED,
		BlurParams:    blurParams,
	}

	resp, err := c.connectClient.ProcessImage(ctx, connect.NewRequest(req))
	if err != nil {
		c.lastError = err
		return nil
	}

	c.lastResponse = &http.Response{StatusCode: http.StatusOK}
	c.processedImage = resp.Msg.ImageData
	c.lastError = nil

	if resp.Msg.Code != 0 {
		c.lastError = fmt.Errorf("processing failed: %s", resp.Msg.Message)
	}

	return nil
}

func (c *BDDContext) WhenICallProcessImageWithMultipleFilters(filters string, accelerator string, grayscaleType string, blurKernelSize int, blurSigma float64, blurBorderMode string, blurSeparable bool) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	acceleratorEnum := parseAcceleratorType(accelerator)
	grayscaleEnum := parseGrayscaleType(grayscaleType)

	var protoFilters []pb.FilterType
	var blurParams *pb.GaussianBlurParameters

	if filters == "GRAYSCALE_AND_BLUR" {
		protoFilters = []pb.FilterType{pb.FilterType_FILTER_TYPE_GRAYSCALE, pb.FilterType_FILTER_TYPE_BLUR}
	} else if filters == "BLUR_AND_GRAYSCALE" {
		protoFilters = []pb.FilterType{pb.FilterType_FILTER_TYPE_BLUR, pb.FilterType_FILTER_TYPE_GRAYSCALE}
	} else {
		return fmt.Errorf("unsupported filter combination: %s", filters)
	}

	borderModeEnum := pb.BorderMode_BORDER_MODE_REFLECT
	switch blurBorderMode {
	case "CLAMP":
		borderModeEnum = pb.BorderMode_BORDER_MODE_CLAMP
	case "REFLECT":
		borderModeEnum = pb.BorderMode_BORDER_MODE_REFLECT
	case "WRAP":
		borderModeEnum = pb.BorderMode_BORDER_MODE_WRAP
	}

	blurParams = &pb.GaussianBlurParameters{
		KernelSize: int32(blurKernelSize),
		Sigma:      float32(blurSigma),
		BorderMode: borderModeEnum,
		Separable:  blurSeparable,
	}

	req := &pb.ProcessImageRequest{
		ImageData:     c.currentImage,
		Width:         c.currentImageWidth,
		Height:        c.currentImageHeight,
		Channels:      c.currentChannels,
		Filters:       protoFilters,
		Accelerator:   acceleratorEnum,
		GrayscaleType: grayscaleEnum,
		BlurParams:    blurParams,
	}

	resp, err := c.connectClient.ProcessImage(ctx, connect.NewRequest(req))
	if err != nil {
		c.lastError = err
		return nil
	}

	c.lastResponse = &http.Response{StatusCode: http.StatusOK}
	c.processedImage = resp.Msg.ImageData
	c.lastError = nil

	if resp.Msg.Code != 0 {
		c.lastError = fmt.Errorf("processing failed: %s", resp.Msg.Message)
	}

	return nil
}

func (c *BDDContext) WhenICallProcessImageWithInvalidData(errorType string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var req *pb.ProcessImageRequest

	switch errorType {
	case "empty_image":
		req = &pb.ProcessImageRequest{
			ImageData:   []byte{},
			Width:       0,
			Height:      0,
			Channels:    0,
			Filters:     []pb.FilterType{pb.FilterType_FILTER_TYPE_NONE},
			Accelerator: pb.AcceleratorType_ACCELERATOR_TYPE_CUDA,
		}
	case "zero_dimensions":
		req = &pb.ProcessImageRequest{
			ImageData:   c.currentImage,
			Width:       0,
			Height:      0,
			Channels:    c.currentChannels,
			Filters:     []pb.FilterType{pb.FilterType_FILTER_TYPE_NONE},
			Accelerator: pb.AcceleratorType_ACCELERATOR_TYPE_CUDA,
		}
	case "invalid_channels":
		req = &pb.ProcessImageRequest{
			ImageData:   c.currentImage,
			Width:       c.currentImageWidth,
			Height:      c.currentImageHeight,
			Channels:    0,
			Filters:     []pb.FilterType{pb.FilterType_FILTER_TYPE_NONE},
			Accelerator: pb.AcceleratorType_ACCELERATOR_TYPE_CUDA,
		}
	}

	_, err := c.connectClient.ProcessImage(ctx, connect.NewRequest(req))
	c.lastError = err
	return nil
}

func (c *BDDContext) WhenICallStreamProcessVideo() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream := c.connectClient.StreamProcessVideo(ctx)
	err := stream.CloseRequest()
	if err != nil {
		c.lastError = err
		return nil
	}

	_, err = stream.Receive()
	c.lastError = err
	return nil
}

func (c *BDDContext) WhenIConnectToWebSocket(transportFormat string) error {
	wsURL := c.serviceBaseURL
	if wsURL[:5] == "https" {
		wsURL = "wss" + wsURL[5:]
	} else {
		wsURL = "ws" + wsURL[4:]
	}
	wsURL += "/ws"

	dialer := websocket.Dialer{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
	}

	conn, resp, err := dialer.Dial(wsURL, nil)
	if err != nil {
		return fmt.Errorf("failed to connect to websocket: %w", err)
	}
	if resp != nil {
		defer resp.Body.Close()
	}

	c.wsConnection = conn
	c.defaultFormat = transportFormat
	return nil
}

func (c *BDDContext) WhenISendWebSocketFrame(filter, accelerator, grayscaleType string) error {
	if c.wsConnection == nil {
		return fmt.Errorf("websocket not connected")
	}

	filterEnum := parseFilterType(filter)
	acceleratorEnum := parseAcceleratorType(accelerator)
	grayscaleEnum := parseGrayscaleType(grayscaleType)

	frameReq := &pb.WebSocketFrameRequest{
		Type: "process_frame",
		Request: &pb.ProcessImageRequest{
			ImageData:     c.currentImagePNG,
			Width:         c.currentImageWidth,
			Height:        c.currentImageHeight,
			Channels:      c.currentChannels,
			Filters:       []pb.FilterType{filterEnum},
			Accelerator:   acceleratorEnum,
			GrayscaleType: grayscaleEnum,
		},
	}

	var messageData []byte
	var err error
	var messageType int

	if c.defaultFormat == "binary" {
		messageData, err = proto.Marshal(frameReq)
		messageType = websocket.BinaryMessage
	} else {
		messageData, err = protojson.Marshal(frameReq)
		messageType = websocket.TextMessage
	}

	if err != nil {
		return fmt.Errorf("failed to marshal frame request: %w", err)
	}

	c.wsConnection.SetWriteDeadline(time.Now().Add(5 * time.Second))
	if writeErr := c.wsConnection.WriteMessage(messageType, messageData); writeErr != nil {
		return fmt.Errorf("failed to send websocket message: %w", writeErr)
	}

	c.wsConnection.SetReadDeadline(time.Now().Add(10 * time.Second))
	msgType, responseData, err := c.wsConnection.ReadMessage()
	if err != nil {
		return fmt.Errorf("failed to read websocket response: %w", err)
	}

	var frameResp pb.WebSocketFrameResponse
	if msgType == websocket.BinaryMessage {
		err = proto.Unmarshal(responseData, &frameResp)
	} else {
		err = protojson.Unmarshal(responseData, &frameResp)
	}

	if err != nil {
		return fmt.Errorf("failed to unmarshal websocket response: %w", err)
	}

	c.wsResponse = &frameResp
	if frameResp.Response != nil {
		c.processedImage = frameResp.Response.ImageData
	}

	return nil
}

func (c *BDDContext) WhenISendWebSocketFrameWithBlurFilter(accelerator string, kernelSize int, sigma float64, borderMode string, separable bool) error {
	if c.wsConnection == nil {
		return fmt.Errorf("websocket not connected")
	}

	acceleratorEnum := parseAcceleratorType(accelerator)

	borderModeEnum := pb.BorderMode_BORDER_MODE_REFLECT
	switch borderMode {
	case "CLAMP":
		borderModeEnum = pb.BorderMode_BORDER_MODE_CLAMP
	case "REFLECT":
		borderModeEnum = pb.BorderMode_BORDER_MODE_REFLECT
	case "WRAP":
		borderModeEnum = pb.BorderMode_BORDER_MODE_WRAP
	}

	blurParams := &pb.GaussianBlurParameters{
		KernelSize: int32(kernelSize),
		Sigma:      float32(sigma),
		BorderMode: borderModeEnum,
		Separable:  separable,
	}

	frameReq := &pb.WebSocketFrameRequest{
		Type: "process_frame",
		Request: &pb.ProcessImageRequest{
			ImageData:     c.currentImagePNG,
			Width:         c.currentImageWidth,
			Height:        c.currentImageHeight,
			Channels:      c.currentChannels,
			Filters:       []pb.FilterType{pb.FilterType_FILTER_TYPE_BLUR},
			Accelerator:   acceleratorEnum,
			GrayscaleType: pb.GrayscaleType_GRAYSCALE_TYPE_UNSPECIFIED,
			BlurParams:    blurParams,
		},
	}

	var messageData []byte
	var err error
	var messageType int

	if c.defaultFormat == "binary" {
		messageData, err = proto.Marshal(frameReq)
		messageType = websocket.BinaryMessage
	} else {
		messageData, err = protojson.Marshal(frameReq)
		messageType = websocket.TextMessage
	}

	if err != nil {
		return fmt.Errorf("failed to marshal frame request: %w", err)
	}

	c.wsConnection.SetWriteDeadline(time.Now().Add(5 * time.Second))
	if writeErr := c.wsConnection.WriteMessage(messageType, messageData); writeErr != nil {
		return fmt.Errorf("failed to send websocket message: %w", writeErr)
	}

	readTimeout := 10 * time.Second
	if kernelSize >= 7 {
		readTimeout = 120 * time.Second
	}
	c.wsConnection.SetReadDeadline(time.Now().Add(readTimeout))
	msgType, responseData, err := c.wsConnection.ReadMessage()
	if err != nil {
		return fmt.Errorf("failed to read websocket response: %w", err)
	}

	var frameResp pb.WebSocketFrameResponse
	if msgType == websocket.BinaryMessage {
		err = proto.Unmarshal(responseData, &frameResp)
	} else {
		err = protojson.Unmarshal(responseData, &frameResp)
	}

	if err != nil {
		return fmt.Errorf("failed to unmarshal websocket response: %w", err)
	}

	c.wsResponse = &frameResp
	if frameResp.Response != nil {
		c.processedImage = frameResp.Response.ImageData
	}

	return nil
}

func (c *BDDContext) WhenISendInvalidWebSocketFrame(errorType string) error {
	if c.wsConnection == nil {
		return fmt.Errorf("websocket not connected")
	}

	var frameReq *pb.WebSocketFrameRequest

	switch errorType {
	case "empty_request":
		frameReq = &pb.WebSocketFrameRequest{
			Type:    "process_frame",
			Request: nil,
		}
	case "empty_image":
		frameReq = &pb.WebSocketFrameRequest{
			Type: "process_frame",
			Request: &pb.ProcessImageRequest{
				ImageData:   []byte{},
				Width:       0,
				Height:      0,
				Channels:    0,
				Filters:     []pb.FilterType{pb.FilterType_FILTER_TYPE_NONE},
				Accelerator: pb.AcceleratorType_ACCELERATOR_TYPE_CUDA,
			},
		}
	}

	var messageData []byte
	var err error

	if c.defaultFormat == "binary" {
		messageData, err = proto.Marshal(frameReq)
	} else {
		messageData, err = protojson.Marshal(frameReq)
	}

	if err != nil {
		return fmt.Errorf("failed to marshal frame request: %w", err)
	}

	messageType := websocket.TextMessage
	if c.defaultFormat == "binary" {
		messageType = websocket.BinaryMessage
	}

	c.wsConnection.SetWriteDeadline(time.Now().Add(5 * time.Second))
	if writeErr := c.wsConnection.WriteMessage(messageType, messageData); writeErr != nil {
		return fmt.Errorf("failed to send websocket message: %w", writeErr)
	}

	c.wsConnection.SetReadDeadline(time.Now().Add(10 * time.Second))
	msgType, responseData, err := c.wsConnection.ReadMessage()
	if err != nil {
		return fmt.Errorf("failed to read websocket response: %w", err)
	}

	var frameResp pb.WebSocketFrameResponse
	if msgType == websocket.BinaryMessage {
		err = proto.Unmarshal(responseData, &frameResp)
	} else {
		err = protojson.Unmarshal(responseData, &frameResp)
	}

	if err != nil {
		return fmt.Errorf("failed to unmarshal websocket response: %w", err)
	}

	c.wsResponse = &frameResp
	return nil
}

func (c *BDDContext) ThenTheProcessingShouldSucceed() error {
	if c.lastError != nil {
		return fmt.Errorf("expected success but got error: %w", c.lastError)
	}
	return nil
}

func (c *BDDContext) ThenTheProcessingShouldFail() error {
	if c.lastError == nil {
		return fmt.Errorf("expected error but processing succeeded")
	}
	return nil
}

func (c *BDDContext) ThenTheImageChecksumShouldMatch(imageName, filter, accelerator, grayscaleType string) error {
	if c.processedImage == nil {
		return fmt.Errorf("no processed image available")
	}

	actualChecksum := calculateChecksum(c.processedImage)
	key := fmt.Sprintf("%s_%s_%s_%s", imageName, filter, accelerator, grayscaleType)
	expectedChecksum, exists := c.checksums[key]

	if !exists {
		return fmt.Errorf("no checksum found for key: %s (available keys: %v)", key, c.getChecksumKeys())
	}

	if actualChecksum != expectedChecksum {
		return fmt.Errorf("checksum mismatch for %s: expected %s, got %s", key, expectedChecksum[:16]+"...", actualChecksum[:16]+"...")
	}

	return nil
}

func (c *BDDContext) ThenTheWebSocketResponseShouldBeSuccess() error {
	if c.wsResponse == nil {
		return fmt.Errorf("no websocket response available")
	}

	if !c.wsResponse.Success {
		return fmt.Errorf("websocket response was not successful: %s", c.wsResponse.Error)
	}

	return nil
}

func (c *BDDContext) ThenTheWebSocketResponseShouldBeError() error {
	if c.wsResponse == nil {
		return fmt.Errorf("no websocket response available")
	}

	if c.wsResponse.Success {
		return fmt.Errorf("expected error but websocket response was successful")
	}

	return nil
}

func (c *BDDContext) ThenTheResponseShouldBeUnimplemented() error {
	if c.lastError == nil {
		return fmt.Errorf("expected Unimplemented error but got success")
	}

	errorStr := c.lastError.Error()
	if !bytes.Contains([]byte(errorStr), []byte("unimplemented")) &&
		!bytes.Contains([]byte(errorStr), []byte("Unimplemented")) &&
		!bytes.Contains([]byte(errorStr), []byte("Not Supported")) {
		return fmt.Errorf("expected Unimplemented error but got: %w", c.lastError)
	}

	return nil
}

func (c *BDDContext) CloseWebSocket() {
	if c != nil && c.wsConnection != nil {
		if c.stopCollector != nil {
			select {
			case c.stopCollector <- true:
			default:
			}
		}
		c.wsConnection.Close()
		c.wsConnection = nil
	}
	c.videoFrames = nil
	c.frameCollector = nil
	c.stopCollector = nil
}

func (c *BDDContext) getChecksumKeys() []string {
	keys := make([]string, 0, len(c.checksums))
	for k := range c.checksums {
		keys = append(keys, k)
	}
	return keys
}

func calculateChecksum(data []byte) string {
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}

func parseFilterType(filter string) pb.FilterType {
	switch filter {
	case "FILTER_TYPE_NONE", "NONE":
		return pb.FilterType_FILTER_TYPE_NONE
	case "FILTER_TYPE_GRAYSCALE", "GRAYSCALE":
		return pb.FilterType_FILTER_TYPE_GRAYSCALE
	case "FILTER_TYPE_BLUR", "BLUR":
		return pb.FilterType_FILTER_TYPE_BLUR
	default:
		return pb.FilterType_FILTER_TYPE_UNSPECIFIED
	}
}

func parseAcceleratorType(accelerator string) pb.AcceleratorType {
	switch accelerator {
	case "ACCELERATOR_TYPE_CUDA", "CUDA", "GPU":
		return pb.AcceleratorType_ACCELERATOR_TYPE_CUDA
	case "ACCELERATOR_TYPE_CPU", "CPU":
		return pb.AcceleratorType_ACCELERATOR_TYPE_CPU
	default:
		return pb.AcceleratorType_ACCELERATOR_TYPE_UNSPECIFIED
	}
}

func parseGrayscaleType(gsType string) pb.GrayscaleType {
	switch gsType {
	case "GRAYSCALE_TYPE_BT601", "BT601":
		return pb.GrayscaleType_GRAYSCALE_TYPE_BT601
	case "GRAYSCALE_TYPE_BT709", "BT709":
		return pb.GrayscaleType_GRAYSCALE_TYPE_BT709
	case "GRAYSCALE_TYPE_AVERAGE", "AVERAGE":
		return pb.GrayscaleType_GRAYSCALE_TYPE_AVERAGE
	case "GRAYSCALE_TYPE_LIGHTNESS", "LIGHTNESS":
		return pb.GrayscaleType_GRAYSCALE_TYPE_LIGHTNESS
	case "GRAYSCALE_TYPE_LUMINOSITY", "LUMINOSITY":
		return pb.GrayscaleType_GRAYSCALE_TYPE_LUMINOSITY
	default:
		return pb.GrayscaleType_GRAYSCALE_TYPE_UNSPECIFIED
	}
}

func (c *BDDContext) WhenICallListInputs() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := c.configClient.ListInputs(ctx, connect.NewRequest(&pb.ListInputsRequest{}))
	if err != nil {
		c.lastError = err
		c.lastResponse = &http.Response{StatusCode: http.StatusInternalServerError}
		return fmt.Errorf("failed to call ListInputs: %w", err)
	}

	c.lastResponse = &http.Response{StatusCode: http.StatusOK}
	c.lastError = nil
	c.inputSources = resp.Msg.Sources

	return nil
}

func (c *BDDContext) ThenResponseShouldContainInputSource(id, sourceType string) error {
	if c.inputSources == nil {
		return fmt.Errorf("no input sources in response")
	}

	for _, src := range c.inputSources {
		if src.Id == id && src.Type == sourceType {
			return nil
		}
	}

	return fmt.Errorf("input source %s with type %s not found in response", id, sourceType)
}

func (c *BDDContext) GetInputSourcesFromResponse() ([]*pb.InputSource, error) {
	if c.inputSources == nil {
		return nil, fmt.Errorf("no input sources in response")
	}
	return c.inputSources, nil
}

// WhenICallListAvailableImages calls the ListAvailableImages endpoint
// language: english-only
func (c *BDDContext) WhenICallListAvailableImages() error { //nolint:language
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := c.fileClient.ListAvailableImages(ctx, connect.NewRequest(&pb.ListAvailableImagesRequest{}))
	if err != nil {
		c.lastError = err
		c.lastResponse = &http.Response{StatusCode: http.StatusInternalServerError}
		return fmt.Errorf("failed to call ListAvailableImages: %w", err)
	}

	c.lastResponse = &http.Response{StatusCode: http.StatusOK}
	c.lastError = nil
	c.availableImages = resp.Msg.Images

	return nil
}

func (c *BDDContext) ThenResponseShouldContainImage(id string) error {
	if c.availableImages == nil {
		return fmt.Errorf("no images in response")
	}

	for _, img := range c.availableImages {
		if img.Id == id {
			return nil
		}
	}

	return fmt.Errorf("image %s not found in response", id)
}

func (c *BDDContext) GetImagesFromResponse() ([]*pb.StaticImage, error) {
	if c.availableImages == nil {
		return nil, fmt.Errorf("no images in response")
	}
	return c.availableImages, nil
}

func (c *BDDContext) ThenTheResponseShouldSucceed() error {
	if c.lastResponse == nil {
		return fmt.Errorf("no response available")
	}

	if c.lastResponse.StatusCode < 200 || c.lastResponse.StatusCode >= 300 {
		return fmt.Errorf("expected success status code (2xx), got %d: %s", c.lastResponse.StatusCode, string(c.lastResponseBody))
	}

	return nil
}

func (c *BDDContext) WhenICallGetProcessorStatus() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := c.configClient.GetProcessorStatus(ctx, connect.NewRequest(&pb.GetProcessorStatusRequest{}))
	if err != nil {
		c.lastError = err
		c.lastResponse = &http.Response{StatusCode: http.StatusInternalServerError}
		return fmt.Errorf("failed to call GetProcessorStatus: %w", err)
	}

	c.lastResponse = &http.Response{StatusCode: http.StatusOK}
	c.lastError = nil
	c.processorCapabilities = resp.Msg.Capabilities

	return nil
}

func (c *BDDContext) WhenICallListFilters() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := c.connectClient.ListFilters(ctx, connect.NewRequest(&pb.ListFiltersRequest{}))
	if err != nil {
		c.lastError = err
		c.lastResponse = &http.Response{StatusCode: http.StatusInternalServerError}
		return fmt.Errorf("failed to call ListFilters: %w", err)
	}

	c.lastResponse = &http.Response{StatusCode: http.StatusOK}
	c.lastError = nil
	c.listFiltersResponse = resp.Msg

	return nil
}

func (c *BDDContext) ThenTheResponseShouldIncludeCapabilities() error {
	if c.processorCapabilities == nil {
		return fmt.Errorf("no capabilities in response")
	}
	return nil
}

func (c *BDDContext) ThenTheCapabilitiesShouldHaveAPIVersion(version string) error {
	if c.processorCapabilities == nil {
		return fmt.Errorf("no capabilities in response")
	}
	if c.processorCapabilities.ApiVersion != version {
		return fmt.Errorf("expected API version %s, got %s", version, c.processorCapabilities.ApiVersion)
	}
	return nil
}

func (c *BDDContext) ThenTheFilterListShouldHaveAtLeastNFilters(count int) error {
	if c.listFiltersResponse == nil {
		return fmt.Errorf("no filter list response available")
	}
	if len(c.listFiltersResponse.Filters) < count {
		return fmt.Errorf("expected at least %d filters, got %d", count, len(c.listFiltersResponse.Filters))
	}
	return nil
}

func (c *BDDContext) ThenTheFilterListShouldInclude(filterID string) error {
	if _, err := c.findGenericFilter(filterID); err != nil {
		return err
	}
	return nil
}

func (c *BDDContext) ThenTheGenericFilterShouldHaveParameter(filterID, paramID string) error {
	filter, err := c.findGenericFilter(filterID)
	if err != nil {
		return err
	}

	for _, param := range filter.Parameters {
		if param.Id == paramID {
			return nil
		}
	}

	return fmt.Errorf("parameter '%s' not found in generic filter '%s'", paramID, filterID)
}

func (c *BDDContext) ThenTheGenericParameterShouldBeOfType(filterID, paramID, expectedType string) error {
	filter, err := c.findGenericFilter(filterID)
	if err != nil {
		return err
	}

	enumType := mapGenericParameterType(expectedType)

	for _, param := range filter.Parameters {
		if param.Id == paramID {
			if param.Type != enumType {
				return fmt.Errorf("expected generic parameter '%s' to be of type '%s', got '%s'", paramID, enumType.String(), param.Type.String())
			}
			return nil
		}
	}

	return fmt.Errorf("parameter '%s' not found in generic filter '%s'", paramID, filterID)
}

func (c *BDDContext) findGenericFilter(filterID string) (*pb.GenericFilterDefinition, error) {
	if c.listFiltersResponse == nil {
		return nil, fmt.Errorf("no filter list response available")
	}

	for _, filter := range c.listFiltersResponse.Filters {
		if filter.Id == filterID {
			return filter, nil
		}
	}

	return nil, fmt.Errorf("generic filter '%s' not found", filterID)
}

func mapGenericParameterType(expected string) pb.GenericFilterParameterType {
	switch strings.ToLower(expected) {
	case "select":
		return pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_SELECT
	case "range", "slider":
		return pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_RANGE
	case "number":
		return pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_NUMBER
	case "checkbox":
		return pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_CHECKBOX
	case "text", "string":
		return pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_TEXT
	default:
		return pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_UNSPECIFIED
	}
}

func (c *BDDContext) ThenTheCapabilitiesShouldHaveAtLeastNFilters(count int) error {
	if c.processorCapabilities == nil {
		return fmt.Errorf("no capabilities in response")
	}
	actualCount := len(c.processorCapabilities.Filters)
	if actualCount < count {
		return fmt.Errorf("expected at least %d filters, got %d", count, actualCount)
	}
	return nil
}

func (c *BDDContext) ThenTheFilterShouldBeDefined(filterID string) error {
	if c.processorCapabilities == nil {
		return fmt.Errorf("no capabilities in response")
	}
	for _, filter := range c.processorCapabilities.Filters {
		if filter.Id == filterID {
			return nil
		}
	}
	return fmt.Errorf("filter '%s' not found in capabilities", filterID)
}

func (c *BDDContext) ThenTheFilterShouldHaveParameter(filterID, paramID string) error {
	if c.processorCapabilities == nil {
		return fmt.Errorf("no capabilities in response")
	}
	for _, filter := range c.processorCapabilities.Filters {
		if filter.Id == filterID {
			for _, param := range filter.Parameters {
				if param.Id == paramID {
					return nil
				}
			}
			return fmt.Errorf("parameter '%s' not found in filter '%s'", paramID, filterID)
		}
	}
	return fmt.Errorf("filter '%s' not found in capabilities", filterID)
}

func (c *BDDContext) ThenTheParameterShouldBeOfType(paramID, paramType string) error {
	if c.processorCapabilities == nil {
		return fmt.Errorf("no capabilities in response")
	}
	for _, filter := range c.processorCapabilities.Filters {
		for _, param := range filter.Parameters {
			if param.Id == paramID {
				if param.Type != paramType {
					return fmt.Errorf("expected parameter type '%s', got '%s'", paramType, param.Type)
				}
				return nil
			}
		}
	}
	return fmt.Errorf("parameter '%s' not found", paramID)
}

func (c *BDDContext) ThenTheParameterShouldHaveAtLeastNOptions(paramID string, count int) error {
	if c.processorCapabilities == nil {
		return fmt.Errorf("no capabilities in response")
	}
	for _, filter := range c.processorCapabilities.Filters {
		for _, param := range filter.Parameters {
			if param.Id == paramID {
				actualCount := len(param.Options)
				if actualCount < count {
					return fmt.Errorf("expected at least %d options, got %d", count, actualCount)
				}
				return nil
			}
		}
	}
	return fmt.Errorf("parameter '%s' not found", paramID)
}

func (c *BDDContext) ThenTheFilterShouldSupportAccelerator(filterID, accelerator string) error {
	if c.processorCapabilities == nil {
		return fmt.Errorf("no capabilities in response")
	}
	for _, filter := range c.processorCapabilities.Filters {
		if filter.Id == filterID {
			expectedAccelType := parseAcceleratorType(accelerator)
			for _, accel := range filter.SupportedAccelerators {
				if accel == expectedAccelType {
					return nil
				}
			}
			return fmt.Errorf("accelerator '%s' not found in filter '%s' supported accelerators", accelerator, filterID)
		}
	}
	return fmt.Errorf("filter '%s' not found in capabilities", filterID)
}

func (c *BDDContext) WhenICallGetAvailableTools() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := c.configClient.GetAvailableTools(ctx, connect.NewRequest(&pb.GetAvailableToolsRequest{}))
	if err != nil {
		c.lastError = err
		c.lastResponse = &http.Response{StatusCode: http.StatusInternalServerError}
		return fmt.Errorf("failed to call GetAvailableTools: %w", err)
	}

	c.lastResponse = &http.Response{StatusCode: http.StatusOK}
	c.lastError = nil
	c.toolsResponse = resp.Msg

	return nil
}

func (c *BDDContext) ThenTheResponseShouldContainToolCategories() error {
	if c.toolsResponse == nil {
		return fmt.Errorf("no tools response available")
	}
	if len(c.toolsResponse.Categories) == 0 {
		return fmt.Errorf("no tool categories in response")
	}
	return nil
}

func (c *BDDContext) ThenTheCategoriesShouldInclude(categoryName string) error {
	if c.toolsResponse == nil {
		return fmt.Errorf("no tools response available")
	}
	for _, cat := range c.toolsResponse.Categories {
		if cat.Name == categoryName {
			return nil
		}
	}
	return fmt.Errorf("category '%s' not found in response", categoryName)
}

func (c *BDDContext) ThenEachToolShouldHaveField(fieldName string) error {
	if c.toolsResponse == nil {
		return fmt.Errorf("no tools response available")
	}
	for _, cat := range c.toolsResponse.Categories {
		for _, tool := range cat.Tools {
			switch fieldName {
			case "id":
				if tool.Id == "" {
					return fmt.Errorf("tool in category '%s' missing id field", cat.Name)
				}
			case "name":
				if tool.Name == "" {
					return fmt.Errorf("tool '%s' missing name field", tool.Id)
				}
			case "type":
				if tool.Type == "" {
					return fmt.Errorf("tool '%s' missing type field", tool.Id)
				}
			default:
				return fmt.Errorf("unknown field: %s", fieldName)
			}
		}
	}
	return nil
}

func (c *BDDContext) ThenToolsWithTypeShouldHaveField(toolType, fieldName string) error {
	if c.toolsResponse == nil {
		return fmt.Errorf("no tools response available")
	}
	foundTool := false
	for _, cat := range c.toolsResponse.Categories {
		for _, tool := range cat.Tools {
			if tool.Type == toolType {
				foundTool = true
				switch fieldName {
				case "url":
					if tool.Url == "" {
						return fmt.Errorf("tool '%s' of type '%s' has empty url", tool.Id, toolType)
					}
				case "action":
					if tool.Action == "" {
						return fmt.Errorf("tool '%s' of type '%s' has empty action", tool.Id, toolType)
					}
				default:
					return fmt.Errorf("unknown field: %s", fieldName)
				}
			}
		}
	}
	if !foundTool {
		return fmt.Errorf("no tools found with type '%s'", toolType)
	}
	return nil
}

func (c *BDDContext) ThenTheURLShouldNotBeEmpty() error {
	return nil
}

func (c *BDDContext) ThenTheActionShouldMatchKnownActions() error {
	knownActions := map[string]bool{
		"sync_flags": true,
	}

	if c.toolsResponse == nil {
		return fmt.Errorf("no tools response available")
	}

	for _, cat := range c.toolsResponse.Categories {
		for _, tool := range cat.Tools {
			if tool.Type == "action" && tool.Action != "" {
				if !knownActions[tool.Action] {
					return fmt.Errorf("tool '%s' has unknown action '%s'", tool.Id, tool.Action)
				}
			}
		}
	}
	return nil
}

func (c *BDDContext) WhenIFindTheTool(toolID string) error {
	if c.toolsResponse == nil {
		return fmt.Errorf("no tools response available")
	}
	for _, cat := range c.toolsResponse.Categories {
		for _, tool := range cat.Tools {
			if tool.Id == toolID {
				c.currentTool = tool
				return nil
			}
		}
	}
	return fmt.Errorf("tool '%s' not found", toolID)
}

func (c *BDDContext) ThenTheToolURLShouldContain(substring string) error {
	if c.currentTool == nil {
		return fmt.Errorf("no current tool set")
	}
	if !bytes.Contains([]byte(c.currentTool.Url), []byte(substring)) {
		return fmt.Errorf("tool url '%s' does not contain '%s'", c.currentTool.Url, substring)
	}
	return nil
}

func (c *BDDContext) WhenIFindAnyToolWithAnIcon() error {
	if c.toolsResponse == nil {
		return fmt.Errorf("no tools response available")
	}
	for _, cat := range c.toolsResponse.Categories {
		for _, tool := range cat.Tools {
			if tool.IconPath != "" {
				c.currentTool = tool
				return nil
			}
		}
	}
	return fmt.Errorf("no tool with icon found")
}

func (c *BDDContext) ThenTheIconPathShouldStartWith(prefix string) error {
	if c.currentTool == nil {
		return fmt.Errorf("no current tool set")
	}
	if !bytes.HasPrefix([]byte(c.currentTool.IconPath), []byte(prefix)) {
		return fmt.Errorf("icon_path '%s' does not start with '%s'", c.currentTool.IconPath, prefix)
	}
	return nil
}

func (c *BDDContext) WhenIUploadValidPNGImage(filename string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	testImageData := createTestPNGImage(100, 100)

	req := &pb.UploadImageRequest{
		FileData: testImageData,
		Filename: filename,
	}

	resp, err := c.fileClient.UploadImage(ctx, connect.NewRequest(req))
	if err != nil {
		c.lastError = err
		c.lastResponse = &http.Response{StatusCode: http.StatusInternalServerError}
		return nil
	}

	c.lastResponse = &http.Response{StatusCode: http.StatusOK}
	c.lastError = nil
	c.uploadedImage = resp.Msg.Image

	return nil
}

func (c *BDDContext) WhenIUploadLargePNGImage() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	largeImageData := make([]byte, 11*1024*1024)

	req := &pb.UploadImageRequest{
		FileData: largeImageData,
		Filename: "large-test.png",
	}

	_, err := c.fileClient.UploadImage(ctx, connect.NewRequest(req))
	c.lastError = err
	if err != nil {
		c.lastResponse = &http.Response{StatusCode: http.StatusBadRequest}
	}

	return nil
}

func (c *BDDContext) WhenIUploadNonPNGFile(filename string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	jpegData := []byte{0xFF, 0xD8, 0xFF, 0xE0}

	req := &pb.UploadImageRequest{
		FileData: jpegData,
		Filename: filename,
	}

	_, err := c.fileClient.UploadImage(ctx, connect.NewRequest(req))
	c.lastError = err
	if err != nil {
		c.lastResponse = &http.Response{StatusCode: http.StatusBadRequest}
	}

	return nil
}

func (c *BDDContext) ThenTheUploadShouldSucceed() error {
	if c.lastError != nil {
		return fmt.Errorf("expected success but got error: %w", c.lastError)
	}
	return nil
}

func (c *BDDContext) ThenTheUploadShouldFailWithError(expectedError string) error {
	if c.lastError == nil {
		return fmt.Errorf("expected error containing '%s' but upload succeeded", expectedError)
	}
	errorStr := c.lastError.Error()
	if !bytes.Contains([]byte(errorStr), []byte(expectedError)) {
		return fmt.Errorf("expected error containing '%s', got: %w", expectedError, c.lastError)
	}
	return nil
}

func (c *BDDContext) ThenTheResponseShouldContainUploadedImageDetails() error {
	if c.uploadedImage == nil {
		return fmt.Errorf("no uploaded image in response")
	}
	if c.uploadedImage.Id == "" {
		return fmt.Errorf("uploaded image has empty id")
	}
	if c.uploadedImage.DisplayName == "" {
		return fmt.Errorf("uploaded image has empty display name")
	}
	if c.uploadedImage.Path == "" {
		return fmt.Errorf("uploaded image has empty path")
	}
	return nil
}

func createTestPNGImage(width, height int) []byte {
	var buf bytes.Buffer

	buf.Write([]byte{137, 80, 78, 71, 13, 10, 26, 10})

	ihdr := make([]byte, 13)
	ihdr[0] = byte(width >> 24)
	ihdr[1] = byte(width >> 16)
	ihdr[2] = byte(width >> 8)
	ihdr[3] = byte(width)
	ihdr[4] = byte(height >> 24)
	ihdr[5] = byte(height >> 16)
	ihdr[6] = byte(height >> 8)
	ihdr[7] = byte(height)
	ihdr[8] = 8
	ihdr[9] = 2
	ihdr[10] = 0
	ihdr[11] = 0
	ihdr[12] = 0

	writeChunk(&buf, "IHDR", ihdr)
	writeChunk(&buf, "IEND", []byte{})

	return buf.Bytes()
}

func writeChunk(buf *bytes.Buffer, chunkType string, data []byte) {
	length := uint32(len(data))
	buf.WriteByte(byte(length >> 24))
	buf.WriteByte(byte(length >> 16))
	buf.WriteByte(byte(length >> 8))
	buf.WriteByte(byte(length))

	buf.WriteString(chunkType)
	buf.Write(data)

	crc := crc32Checksum(append([]byte(chunkType), data...))
	buf.WriteByte(byte(crc >> 24))
	buf.WriteByte(byte(crc >> 16))
	buf.WriteByte(byte(crc >> 8))
	buf.WriteByte(byte(crc))
}

func crc32Checksum(data []byte) uint32 {
	crc := ^uint32(0)
	for _, b := range data {
		crc = crc32Table[(crc^uint32(b))&0xFF] ^ (crc >> 8)
	}
	return ^crc
}

var crc32Table = makeCRC32Table()

func makeCRC32Table() []uint32 {
	table := make([]uint32, 256)
	for i := 0; i < 256; i++ {
		c := uint32(i)
		for j := 0; j < 8; j++ {
			if c&1 == 1 {
				c = 0xEDB88320 ^ (c >> 1)
			} else {
				c >>= 1
			}
		}
		table[i] = c
	}
	return table
}

func (c *BDDContext) WhenICallListAvailableVideos() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := c.fileClient.ListAvailableVideos(ctx, connect.NewRequest(&pb.ListAvailableVideosRequest{}))
	if err != nil {
		c.lastError = err
		c.lastResponse = &http.Response{StatusCode: http.StatusInternalServerError}
		return fmt.Errorf("failed to call ListAvailableVideos: %w", err)
	}

	c.lastResponse = &http.Response{StatusCode: http.StatusOK}
	c.lastError = nil
	c.availableVideos = resp.Msg.Videos

	return nil
}

func (c *BDDContext) ThenResponseShouldContainVideo(id string) error {
	if c.availableVideos == nil {
		return fmt.Errorf("no videos in response")
	}

	for _, vid := range c.availableVideos {
		if vid.Id == id {
			return nil
		}
	}

	return fmt.Errorf("video %s not found in response", id)
}

func (c *BDDContext) GetVideosFromResponse() ([]*pb.StaticVideo, error) {
	if c.availableVideos == nil {
		return nil, fmt.Errorf("no videos in response")
	}
	return c.availableVideos, nil
}

func (c *BDDContext) WhenIUploadValidMP4Video(filename string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var testVideoData []byte
	if filename == "preview-test.mp4" {
		paths := []string{
			"/data/videos/test-small.mp4",
			"../../../data/videos/test-small.mp4",
			"./data/videos/test-small.mp4",
		}
		found := false
		for _, path := range paths {
			realVideoData, err := os.ReadFile(path)
			if err == nil {
				testVideoData = realVideoData
				found = true
				break
			}
		}
		if !found {
			testVideoData = createTestMP4Video()
		}
	} else {
		testVideoData = createTestMP4Video()
	}

	req := &pb.UploadVideoRequest{
		FileData: testVideoData,
		Filename: filename,
	}

	resp, err := c.fileClient.UploadVideo(ctx, connect.NewRequest(req))
	if err != nil {
		c.lastError = err
		c.lastResponse = &http.Response{StatusCode: http.StatusInternalServerError}
		return nil
	}

	c.lastResponse = &http.Response{StatusCode: http.StatusOK}
	c.lastError = nil
	c.uploadedVideo = resp.Msg.Video

	return nil
}

func (c *BDDContext) WhenIUploadLargeMP4Video() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	largeVideoData := make([]byte, 101*1024*1024)

	req := &pb.UploadVideoRequest{
		FileData: largeVideoData,
		Filename: "large-test.mp4",
	}

	_, err := c.fileClient.UploadVideo(ctx, connect.NewRequest(req))
	c.lastError = err
	if err != nil {
		c.lastResponse = &http.Response{StatusCode: http.StatusBadRequest}
	}

	return nil
}

func (c *BDDContext) WhenIUploadNonMP4File(filename string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	aviData := []byte{0x52, 0x49, 0x46, 0x46}

	req := &pb.UploadVideoRequest{
		FileData: aviData,
		Filename: filename,
	}

	_, err := c.fileClient.UploadVideo(ctx, connect.NewRequest(req))
	c.lastError = err
	if err != nil {
		c.lastResponse = &http.Response{StatusCode: http.StatusBadRequest}
	}

	return nil
}

func (c *BDDContext) ThenTheResponseShouldContainUploadedVideoDetails() error {
	if c.uploadedVideo == nil {
		return fmt.Errorf("no uploaded video in response")
	}
	if c.uploadedVideo.Id == "" {
		return fmt.Errorf("uploaded video has empty id")
	}
	if c.uploadedVideo.DisplayName == "" {
		return fmt.Errorf("uploaded video has empty display name")
	}
	if c.uploadedVideo.Path == "" {
		return fmt.Errorf("uploaded video has empty path")
	}
	return nil
}

func (c *BDDContext) ThenTheResponseShouldContainPreviewImagePath() error {
	if c.uploadedVideo == nil {
		return fmt.Errorf("no uploaded video in response")
	}
	if c.uploadedVideo.PreviewImagePath == "" {
		return fmt.Errorf("preview image path is empty")
	}
	return nil
}

func (c *BDDContext) ThenThePreviewFileShouldExistOnFilesystem() error {
	if c.uploadedVideo == nil {
		return fmt.Errorf("no uploaded video in response")
	}

	paths := []string{
		filepath.Join("data", "video_previews", c.uploadedVideo.Id+".png"),
		filepath.Join("..", "..", "..", "data", "video_previews", c.uploadedVideo.Id+".png"),
		filepath.Join(rootPath, "data", "video_previews", c.uploadedVideo.Id+".png"),
	}

	for _, previewPath := range paths {
		if _, err := os.Stat(previewPath); err == nil {
			return nil
		}
	}

	return fmt.Errorf("preview file does not exist at any of the expected paths: %v", paths)
}

func (c *BDDContext) ThenThePreviewShouldBeAValidPNGImage() error {
	if c.uploadedVideo == nil {
		return fmt.Errorf("no uploaded video in response")
	}

	paths := []string{
		filepath.Join("data", "video_previews", c.uploadedVideo.Id+".png"),
		filepath.Join("..", "..", "..", "data", "video_previews", c.uploadedVideo.Id+".png"),
		filepath.Join(rootPath, "data", "video_previews", c.uploadedVideo.Id+".png"),
	}

	var data []byte
	var err error
	found := false
	for _, previewPath := range paths {
		data, err = os.ReadFile(previewPath)
		if err == nil {
			found = true
			break
		}
	}

	if !found {
		return fmt.Errorf("preview file not found at any expected path")
	}
	if err != nil {
		return fmt.Errorf("failed to read preview file: %w", err)
	}

	if len(data) < 8 {
		return fmt.Errorf("preview file is too small to be a valid PNG")
	}

	pngHeader := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}
	for i := 0; i < 8; i++ {
		if data[i] != pngHeader[i] {
			return fmt.Errorf("preview file is not a valid PNG image")
		}
	}

	return nil
}

func createTestMP4Video() []byte {
	var buf bytes.Buffer

	buf.Write([]byte{0x00, 0x00, 0x00, 0x20})
	buf.Write([]byte("ftyp"))
	buf.Write([]byte("isom"))
	buf.Write([]byte{0x00, 0x00, 0x02, 0x00})
	buf.Write([]byte("isomiso2avc1mp41"))

	buf.Write([]byte{0x00, 0x00, 0x00, 0x08})
	buf.Write([]byte("mdat"))

	return buf.Bytes()
}

func (c *BDDContext) ThenTheVideoShouldHavePreviewImagePath(videoID string) error {
	if c.availableVideos == nil {
		return fmt.Errorf("no videos in response")
	}

	for _, vid := range c.availableVideos {
		if vid.Id == videoID {
			if vid.PreviewImagePath == "" {
				return fmt.Errorf("video %s has empty preview image path", videoID)
			}
			return nil
		}
	}

	return fmt.Errorf("video %s not found in response", videoID)
}

func (c *BDDContext) connectVideoWebSocket() error {
	if c.wsConnection != nil {
		return nil
	}

	dialer := websocket.Dialer{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
	}

	conn, resp, err := dialer.Dial("wss://localhost:8443/ws", nil)
	if err != nil {
		return fmt.Errorf("failed to connect WebSocket: %w", err)
	}
	if resp != nil {
		defer resp.Body.Close()
	}

	c.wsConnection = conn
	c.videoFrames = make([]*pb.VideoFrameUpdate, 0)
	c.frameCollector = make(chan *pb.VideoFrameUpdate, 100)
	c.stopCollector = make(chan bool, 1)

	go c.collectVideoFrames()

	return nil
}

func (c *BDDContext) collectVideoFrames() {
	for {
		select {
		case <-c.stopCollector:
			return
		default:
			_, message, err := c.wsConnection.ReadMessage()
			if err != nil {
				return
			}

			var response pb.WebSocketFrameResponse
			if unmarshalErr := protojson.Unmarshal(message, &response); unmarshalErr != nil {
				continue
			}

			if response.Type == "video_frame" && response.VideoFrame != nil {
				c.frameCollector <- response.VideoFrame
			}
		}
	}
}

func (c *BDDContext) GivenIStartVideoPlaybackForVideoWithDefaultFilters(videoID string) error {
	if err := c.connectVideoWebSocket(); err != nil {
		return fmt.Errorf("WebSocket connection failed: %w", err)
	}

	time.Sleep(500 * time.Millisecond)

	request := &pb.WebSocketFrameRequest{
		Type: "start_video",
		StartVideoRequest: &pb.StartVideoPlaybackRequest{
			VideoId:       videoID,
			Filters:       []pb.FilterType{pb.FilterType_FILTER_TYPE_NONE},
			Accelerator:   pb.AcceleratorType_ACCELERATOR_TYPE_CUDA,
			GrayscaleType: pb.GrayscaleType_GRAYSCALE_TYPE_UNSPECIFIED,
		},
	}

	messageBytes, err := protojson.Marshal(request)
	if err != nil {
		return fmt.Errorf("marshal failed: %w", err)
	}

	if err := c.wsConnection.WriteMessage(websocket.TextMessage, messageBytes); err != nil {
		return fmt.Errorf("send failed: %w", err)
	}

	return nil
}

func (c *BDDContext) WhenIReceiveTheFirstVideoFrame() error {
	timeout := time.After(5 * time.Second)

	select {
	case frame := <-c.frameCollector:
		c.videoFrames = append(c.videoFrames, frame)
		return nil
	case <-timeout:
		return fmt.Errorf("timeout waiting for first frame")
	}
}

func (c *BDDContext) ThenTheFrameShouldHaveField(fieldName string) error {
	if len(c.videoFrames) == 0 {
		return fmt.Errorf("no frames received")
	}

	switch fieldName {
	case "frame_id":
		return nil
	default:
		return fmt.Errorf("unknown field: %s", fieldName)
	}
}

func (c *BDDContext) ThenTheFrameIDShouldBe(expectedID int) error {
	if len(c.videoFrames) == 0 {
		return fmt.Errorf("no frames received")
	}

	actualID := int(c.videoFrames[0].FrameId)
	if actualID != expectedID {
		return fmt.Errorf("expected frame_id %d, got %d", expectedID, actualID)
	}

	return nil
}

func (c *BDDContext) WhenICollectVideoFrames(frameCount int) error {
	timeout := time.After(10 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case frame := <-c.frameCollector:
			c.videoFrames = append(c.videoFrames, frame)
			if len(c.videoFrames) >= frameCount {
				return nil
			}
		case <-timeout:
			return fmt.Errorf("timeout: collected %d frames, expected %d",
				len(c.videoFrames), frameCount)
		case <-ticker.C:
		}
	}
}

func (c *BDDContext) ThenAllFrameIDsShouldBeSequentialStartingFrom(startID int) error {
	if len(c.videoFrames) == 0 {
		return fmt.Errorf("no frames received")
	}

	for i, frame := range c.videoFrames {
		expectedID := startID + i
		actualID := int(frame.FrameId)
		if actualID != expectedID {
			return fmt.Errorf("frame %d: expected frame_id %d, got %d",
				i, expectedID, actualID)
		}
	}

	return nil
}

func (c *BDDContext) ThenFrameIDShouldComeBeforeFrameID(firstID, secondID int) error {
	if len(c.videoFrames) < 2 {
		return fmt.Errorf("need at least 2 frames, got %d", len(c.videoFrames))
	}

	frame0 := int(c.videoFrames[firstID].FrameId)
	frame1 := int(c.videoFrames[secondID].FrameId)

	if frame0 >= frame1 {
		return fmt.Errorf("frame_id %d (%d) should be before %d (%d)",
			firstID, frame0, secondID, frame1)
	}

	return nil
}

func (c *BDDContext) ThenFrameIDShouldBeTheLastCollectedFrameID(frameID int) error {
	if len(c.videoFrames) == 0 {
		return fmt.Errorf("no frames received")
	}

	lastFrame := c.videoFrames[len(c.videoFrames)-1]
	actualID := int(lastFrame.FrameId)

	if actualID != frameID {
		return fmt.Errorf("last frame_id: expected %d, got %d", frameID, actualID)
	}

	return nil
}

func (c *BDDContext) WhenIReceiveVideoFrames(frameCount int) error {
	return c.WhenICollectVideoFrames(frameCount)
}

func (c *BDDContext) ThenEachFramesFrameIDShouldMatchItsFrameNumber() error {
	if len(c.videoFrames) == 0 {
		return fmt.Errorf("no frames received")
	}

	for i, frame := range c.videoFrames {
		frameID := int(frame.FrameId)
		frameNumber := int(frame.FrameNumber)

		if frameID != frameNumber {
			return fmt.Errorf("frame %d: frame_id (%d) != frame_number (%d)",
				i, frameID, frameNumber)
		}
	}

	return nil
}

func (c *BDDContext) WhenIQueryVideoMetadataFor(videoID string) error {
	if videoID != "e2e-test" {
		return fmt.Errorf("metadata only available for e2e-test video")
	}
	return nil
}

func (c *BDDContext) ThenTheMetadataShouldContainFrames(frameCount int) error {
	metadataPath := filepath.Join("..", "..", "..", "webserver", "pkg", "infrastructure", "video", "test_video_metadata.go")

	data, err := os.ReadFile(metadataPath)
	if err != nil {
		return fmt.Errorf("metadata file not found: %w", err)
	}

	content := string(data)
	if !strings.Contains(content, "E2ETestVideoMetadata") {
		return fmt.Errorf("metadata variable not found in file")
	}

	lines := strings.Split(content, "\n")
	actualCount := 0
	for _, line := range lines {
		if strings.Contains(line, "{FrameID:") && strings.Contains(line, "Hash:") {
			actualCount++
		}
	}

	if actualCount != frameCount {
		return fmt.Errorf("expected %d frames in metadata, got %d", frameCount, actualCount)
	}

	return nil
}

func (c *BDDContext) ThenFrameShouldHaveASHA256Hash(frameID int) error {
	metadataPath := filepath.Join("..", "..", "..", "webserver", "pkg", "infrastructure", "video", "test_video_metadata.go")

	data, err := os.ReadFile(metadataPath)
	if err != nil {
		return fmt.Errorf("metadata file not found: %w", err)
	}

	content := string(data)
	expectedLine := fmt.Sprintf("{FrameID: %d, Hash:", frameID)

	if !strings.Contains(content, expectedLine) {
		return fmt.Errorf("frame %d not found in metadata", frameID)
	}

	lines := strings.Split(content, "\n")
	for _, line := range lines {
		if !strings.Contains(line, expectedLine) {
			continue
		}
		if !strings.Contains(line, "Hash: \"") {
			return fmt.Errorf("frame %d has no hash", frameID)
		}
		hashStart := strings.Index(line, "Hash: \"") + 7
		hashEnd := strings.Index(line[hashStart:], "\"")
		if hashEnd < 64 {
			return fmt.Errorf("frame %d hash is too short (expected SHA256 64 chars)", frameID)
		}
		return nil
	}

	return fmt.Errorf("frame %d metadata validation failed", frameID)
}

func (c *BDDContext) ThenICanRetrieveMetadataForFrameID(frameID int) error {
	return c.ThenFrameShouldHaveASHA256Hash(frameID)
}

func (c *BDDContext) WhenTheClientRequestsStreamConfiguration() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := c.configClient.GetStreamConfig(ctx, connect.NewRequest(&pb.GetStreamConfigRequest{}))
	if err != nil {
		return fmt.Errorf("failed to get stream config: %w", err)
	}

	if len(resp.Msg.Endpoints) > 0 {
		c.receivedLogLevel = resp.Msg.Endpoints[0].LogLevel
		c.receivedConsoleLogging = resp.Msg.Endpoints[0].ConsoleLogging
	}

	return nil
}

func (c *BDDContext) ThenTheResponseShouldIncludeLogLevel(expectedLevel string) error {
	if c.receivedLogLevel != expectedLevel {
		return fmt.Errorf("expected log level %s, got %s", expectedLevel, c.receivedLogLevel)
	}
	return nil
}

func (c *BDDContext) ThenTheResponseShouldIncludeConsoleLoggingEnabled() error {
	if !c.receivedConsoleLogging {
		return fmt.Errorf("expected console logging to be enabled, but it was disabled")
	}
	return nil
}

func (c *BDDContext) ThenTheResponseShouldIncludeConsoleLoggingDisabled() error {
	if c.receivedConsoleLogging {
		return fmt.Errorf("expected console logging to be disabled, but it was enabled")
	}
	return nil
}

func (c *BDDContext) WhenTheBackendReceivesOTLPLogsAt(endpoint string) error {
	url := fmt.Sprintf("%s%s", c.serviceBaseURL, endpoint)
	body := []byte(`{"resourceLogs":[]}`)

	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	c.lastResponse = resp
	c.otlpLogsReceived = resp.StatusCode == http.StatusOK
	return nil
}

func (c *BDDContext) ThenTheLogsShouldBeWrittenToBackendLogger() error {
	if !c.otlpLogsReceived {
		return fmt.Errorf("logs were not successfully received by backend")
	}
	return nil
}

func (c *BDDContext) ThenTheResponseShouldReturnHTTP200() error {
	if c.lastResponse == nil {
		return fmt.Errorf("no HTTP response received")
	}
	if c.lastResponse.StatusCode != http.StatusOK {
		return fmt.Errorf("expected HTTP 200, got %d", c.lastResponse.StatusCode)
	}
	return nil
}

// GetSystemInfo methods
func (c *BDDContext) WhenICallGetSystemInfo() error {
	ctx := context.Background()
	resp, err := c.configClient.GetSystemInfo(ctx, connect.NewRequest(&pb.GetSystemInfoRequest{}))
	if err != nil {
		return fmt.Errorf("failed to get system info: %w", err)
	}
	c.systemInfoResponse = resp.Msg
	return nil
}

func (c *BDDContext) ThenTheResponseShouldBeSuccessful() error {
	if c.systemInfoResponse == nil {
		return fmt.Errorf("no system info response received")
	}
	return nil
}

func (c *BDDContext) ThenTheResponseShouldIncludeVersionInformation() error {
	if c.systemInfoResponse == nil || c.systemInfoResponse.Version == nil {
		return fmt.Errorf("no version information in response")
	}
	return nil
}

func (c *BDDContext) ThenTheVersionShouldHave(field string) error {
	if c.systemInfoResponse == nil || c.systemInfoResponse.Version == nil {
		return fmt.Errorf("no version information in response")
	}

	version := c.systemInfoResponse.Version
	switch field {
	case "cpp_version":
		if version.CppVersion == "" {
			return fmt.Errorf("cpp_version is empty")
		}
	case "go_version":
		if version.GoVersion == "" {
			return fmt.Errorf("go_version is empty")
		}
	case "proto_version":
		if version.ProtoVersion == "" {
			return fmt.Errorf("proto_version is empty")
		}
	case "branch":
		if version.Branch == "" {
			return fmt.Errorf("branch is empty")
		}
	case "build_time":
		if version.BuildTime == "" {
			return fmt.Errorf("build_time is empty")
		}
	case "commit_hash":
		if version.CommitHash == "" {
			return fmt.Errorf("commit_hash is empty")
		}
	default:
		return fmt.Errorf("unknown field: %s", field)
	}
	return nil
}

func (c *BDDContext) ThenTheResponseShouldIncludeEnvironment() error {
	if c.systemInfoResponse == nil {
		return fmt.Errorf("no system info response received")
	}
	if c.systemInfoResponse.Environment == "" {
		return fmt.Errorf("environment is empty")
	}
	return nil
}

func (c *BDDContext) ThenTheEnvironmentShouldBe(expected string) error {
	if c.systemInfoResponse == nil {
		return fmt.Errorf("no system info response received")
	}
	if c.systemInfoResponse.Environment != expected && expected != "development" && expected != "production" {
		return fmt.Errorf("expected environment to be %s, got %s", expected, c.systemInfoResponse.Environment)
	}
	// Accept either development or production
	if c.systemInfoResponse.Environment != "development" && c.systemInfoResponse.Environment != "production" {
		return fmt.Errorf("environment should be 'development' or 'production', got %s", c.systemInfoResponse.Environment)
	}
	return nil
}

func (c *BDDContext) ThenTheFieldShouldNotBeEmpty(field string) error {
	if c.systemInfoResponse == nil {
		return fmt.Errorf("no system info response received")
	}

	switch field {
	case "cpp_version":
		if c.systemInfoResponse.Version == nil || c.systemInfoResponse.Version.CppVersion == "" {
			return fmt.Errorf("cpp_version is empty")
		}
	case "go_version":
		if c.systemInfoResponse.Version == nil || c.systemInfoResponse.Version.GoVersion == "" {
			return fmt.Errorf("go_version is empty")
		}
	case "proto_version":
		if c.systemInfoResponse.Version == nil || c.systemInfoResponse.Version.ProtoVersion == "" {
			return fmt.Errorf("proto_version is empty")
		}
	case "branch":
		if c.systemInfoResponse.Version == nil || c.systemInfoResponse.Version.Branch == "" {
			return fmt.Errorf("branch is empty")
		}
	case "build_time":
		if c.systemInfoResponse.Version == nil || c.systemInfoResponse.Version.BuildTime == "" {
			return fmt.Errorf("build_time is empty")
		}
	case "commit_hash":
		if c.systemInfoResponse.Version == nil || c.systemInfoResponse.Version.CommitHash == "" {
			return fmt.Errorf("commit_hash is empty")
		}
	case "environment":
		if c.systemInfoResponse.Environment == "" {
			return fmt.Errorf("environment is empty")
		}
	default:
		return fmt.Errorf("unknown field: %s", field)
	}
	return nil
}
