package featureflags

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type FliptHTTPAPI struct {
	*FliptWriter
}

func NewFliptHTTPAPI(grpcURL, namespace string, client httpClient) *FliptHTTPAPI {
	return &FliptHTTPAPI{
		FliptWriter: NewFliptWriter(grpcURL, namespace, client),
	}
}

type Flag struct {
	Key         string `json:"key"`
	Name        string `json:"name"`
	Type        string `json:"type"`
	Enabled     bool   `json:"enabled"`
	Description string `json:"description"`
}

func (api *FliptHTTPAPI) GetFlag(ctx context.Context, flagKey string) (*Flag, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	url := fmt.Sprintf("%s/api/v1/namespaces/%s/flags/%s", api.apiURL, api.namespace, flagKey)

	req, err := http.NewRequestWithContext(timeoutCtx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return nil, err
	}

	resp, err := api.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil, fmt.Errorf("flag not found")
	}

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body) //nolint:errcheck // Best effort error logging
		return nil, fmt.Errorf("unexpected status code %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var flag Flag
	if err := json.NewDecoder(resp.Body).Decode(&flag); err != nil {
		return nil, fmt.Errorf("failed to decode flag: %w", err)
	}

	return &flag, nil
}

func (api *FliptHTTPAPI) ListFlags(ctx context.Context) ([]Flag, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	url := fmt.Sprintf("%s/api/v1/namespaces/%s/flags", api.apiURL, api.namespace)

	req, err := http.NewRequestWithContext(timeoutCtx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return nil, err
	}

	resp, err := api.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body) //nolint:errcheck // Best effort error logging
		return nil, fmt.Errorf("unexpected status code %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var result struct {
		Flags []Flag `json:"flags"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode flags: %w", err)
	}

	return result.Flags, nil
}

func (api *FliptHTTPAPI) CleanAllFlags(ctx context.Context) error {
	flags, err := api.ListFlags(ctx)
	if err != nil {
		return fmt.Errorf("failed to list flags: %w", err)
	}

	for _, flag := range flags {
		if err := api.DeleteFlag(ctx, flag.Key); err != nil {
			return fmt.Errorf("failed to delete flag %s: %w", flag.Key, err)
		}
	}

	return nil
}
