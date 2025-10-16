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
	_ "image/png"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"connectrpc.com/connect"
	"github.com/gorilla/websocket"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/proto/gen/genconnect"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/featureflags"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

type BDDContext struct {
	fliptAPI           *featureflags.FliptHTTPAPI
	httpClient         *http.Client
	serviceBaseURL     string
	lastResponse       *http.Response
	lastResponseBody   []byte
	defaultFormat      string
	defaultEndpoint    string
	connectClient      genconnect.ImageProcessorServiceClient
	currentImage       []byte
	currentImagePNG    []byte
	currentImageWidth  int32
	currentImageHeight int32
	currentChannels    int32
	processedImage     []byte
	wsConnection       *websocket.Conn
	wsResponse         *pb.WebSocketFrameResponse
	lastError          error
	checksums          map[string]string
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

	ctx := &BDDContext{
		fliptAPI:       featureflags.NewFliptHTTPAPI(fliptBaseURL, fliptNamespace, httpClient),
		httpClient:     httpClient,
		serviceBaseURL: serviceBaseURL,
		connectClient:  connectClient,
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
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
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

func (c *BDDContext) WhenICallGetStreamConfig() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	url := fmt.Sprintf("%s/cuda_learning.ConfigService/GetStreamConfig", c.serviceBaseURL)

	reqBody := bytes.NewBufferString("{}")
	req, err := http.NewRequestWithContext(ctx, "POST", url, reqBody)
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to call GetStreamConfig: %w", err)
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

func (c *BDDContext) WhenICallSyncFeatureFlags() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	url := fmt.Sprintf("%s/cuda_learning.ConfigService/SyncFeatureFlags", c.serviceBaseURL)

	reqBody := bytes.NewBufferString("{}")
	req, err := http.NewRequestWithContext(ctx, "POST", url, reqBody)
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to call SyncFeatureFlags: %w", err)
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

func (c *BDDContext) WhenIWaitForFlagsToBeSynced() error {
	time.Sleep(1 * time.Second)
	return nil
}

func (c *BDDContext) WhenICallHealthEndpoint() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	url := fmt.Sprintf("%s/health", c.serviceBaseURL)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
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

	rawData := make([]byte, width*height*4)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, a := img.At(x, y).RGBA()
			idx := (y*width + x) * 4
			rawData[idx] = byte(r >> 8)
			rawData[idx+1] = byte(g >> 8)
			rawData[idx+2] = byte(b >> 8)
			rawData[idx+3] = byte(a >> 8)
		}
	}

	c.currentImage = rawData
	c.currentImageWidth = int32(width)
	c.currentImageHeight = int32(height)
	c.currentChannels = 4

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

	c.lastResponse = &http.Response{StatusCode: 200}
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
			Accelerator: pb.AcceleratorType_ACCELERATOR_TYPE_GPU,
		}
	case "zero_dimensions":
		req = &pb.ProcessImageRequest{
			ImageData:   c.currentImage,
			Width:       0,
			Height:      0,
			Channels:    c.currentChannels,
			Filters:     []pb.FilterType{pb.FilterType_FILTER_TYPE_NONE},
			Accelerator: pb.AcceleratorType_ACCELERATOR_TYPE_GPU,
		}
	case "invalid_channels":
		req = &pb.ProcessImageRequest{
			ImageData:   c.currentImage,
			Width:       c.currentImageWidth,
			Height:      c.currentImageHeight,
			Channels:    0,
			Filters:     []pb.FilterType{pb.FilterType_FILTER_TYPE_NONE},
			Accelerator: pb.AcceleratorType_ACCELERATOR_TYPE_GPU,
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
	wsURL = wsURL + "/ws"

	dialer := websocket.Dialer{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
	}

	conn, _, err := dialer.Dial(wsURL, nil)
	if err != nil {
		return fmt.Errorf("failed to connect to websocket: %w", err)
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
	if err := c.wsConnection.WriteMessage(messageType, messageData); err != nil {
		return fmt.Errorf("failed to send websocket message: %w", err)
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
				Accelerator: pb.AcceleratorType_ACCELERATOR_TYPE_GPU,
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
	if err := c.wsConnection.WriteMessage(messageType, messageData); err != nil {
		return fmt.Errorf("failed to send websocket message: %w", err)
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
		return fmt.Errorf("expected success but got error: %v", c.lastError)
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
		return fmt.Errorf("expected Unimplemented error but got: %v", c.lastError)
	}

	return nil
}

func (c *BDDContext) CloseWebSocket() {
	if c != nil && c.wsConnection != nil {
		c.wsConnection.Close()
		c.wsConnection = nil
	}
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
	default:
		return pb.FilterType_FILTER_TYPE_UNSPECIFIED
	}
}

func parseAcceleratorType(accelerator string) pb.AcceleratorType {
	switch accelerator {
	case "ACCELERATOR_TYPE_GPU", "GPU":
		return pb.AcceleratorType_ACCELERATOR_TYPE_GPU
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
