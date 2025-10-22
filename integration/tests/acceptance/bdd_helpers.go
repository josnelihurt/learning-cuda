package acceptance

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/featureflags"
)

type BDDContext struct {
	fliptAPI         *featureflags.FliptHTTPAPI
	httpClient       *http.Client
	serviceBaseURL   string
	lastResponse     *http.Response
	lastResponseBody []byte
	defaultFormat    string
	defaultEndpoint  string
}

func NewBDDContext(fliptBaseURL, fliptNamespace, serviceBaseURL string) *BDDContext {
	httpClient := &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: true,
			},
		},
		Timeout: 10 * time.Second,
	}

	return &BDDContext{
		fliptAPI:       featureflags.NewFliptHTTPAPI(fliptBaseURL, fliptNamespace, httpClient),
		httpClient:     httpClient,
		serviceBaseURL: serviceBaseURL,
	}
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
