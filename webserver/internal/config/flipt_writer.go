package config

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"
)

type httpClient interface {
	Do(req *http.Request) (*http.Response, error)
}

type FliptWriter struct {
	apiURL    string
	namespace string
	client    httpClient
}

func NewFliptWriter(grpcURL, namespace string, client httpClient) *FliptWriter {
	restAPIURL := convertGRPCToRESTURL(grpcURL)
	
	return &FliptWriter{
		apiURL:    restAPIURL,
		namespace: namespace,
		client:    client,
	}
}

func convertGRPCToRESTURL(grpcURL string) string {
	restURL := strings.Replace(grpcURL, ":9000", ":8081", 1)
	return restURL
}

func (fw *FliptWriter) SyncFlags(ctx context.Context, flags map[string]interface{}) error {
	failedFlags := []string{}
	
	for key, value := range flags {
		if boolVal, ok := value.(bool); ok {
			if err := fw.ensureFlagExists(ctx, key, boolVal); err != nil {
				log.Printf("Failed to sync flag '%s': %v", key, err)
				failedFlags = append(failedFlags, key)
			}
		}
	}
	
	if len(failedFlags) > 0 {
		return fmt.Errorf("failed to sync %d flags: %v", len(failedFlags), failedFlags)
	}
	
	return nil
}

func (fw *FliptWriter) ensureFlagExists(ctx context.Context, flagKey string, defaultEnabled bool) error {
	exists, err := fw.checkFlagExists(ctx, flagKey)
	if err != nil {
		return fmt.Errorf("failed to check flag existence: %w", err)
	}
	
	if exists {
		return nil
	}
	
	if err := fw.createFlag(ctx, flagKey, defaultEnabled); err != nil {
		return fmt.Errorf("failed to create flag: %w", err)
	}
	
	if err := fw.createBooleanVariant(ctx, flagKey, defaultEnabled); err != nil {
		log.Printf("WARNING: Flag created but variant setup failed for '%s': %v", flagKey, err)
	}
	
	return nil
}

func (fw *FliptWriter) checkFlagExists(ctx context.Context, flagKey string) (bool, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	
	url := fmt.Sprintf("%s/api/v1/namespaces/%s/flags/%s", fw.apiURL, fw.namespace, flagKey)
	req, err := http.NewRequestWithContext(timeoutCtx, "GET", url, nil)
	if err != nil {
		return false, err
	}
	
	resp, err := fw.client.Do(req)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	
	return resp.StatusCode == http.StatusOK, nil
}

func (fw *FliptWriter) createFlag(ctx context.Context, flagKey string, defaultEnabled bool) error {
	timeoutCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	
	url := fmt.Sprintf("%s/api/v1/namespaces/%s/flags", fw.apiURL, fw.namespace)
	
	payload := map[string]interface{}{
		"key":         flagKey,
		"name":        flagKey,
		"description": fmt.Sprintf("Auto-synced from config.yaml (default: %v)", defaultEnabled),
		"enabled":     defaultEnabled,
		"type":        "BOOLEAN_FLAG_TYPE",
	}
	
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	
	req, err := http.NewRequestWithContext(timeoutCtx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	
	resp, err := fw.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}
	
	return nil
}

func (fw *FliptWriter) createBooleanVariant(ctx context.Context, flagKey string, defaultValue bool) error {
	timeoutCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	
	url := fmt.Sprintf("%s/api/v1/namespaces/%s/flags/%s/variants", fw.apiURL, fw.namespace, flagKey)
	
	variantKey := "enabled"
	if !defaultValue {
		variantKey = "disabled"
	}
	
	payload := map[string]interface{}{
		"key": variantKey,
	}
	
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	
	req, err := http.NewRequestWithContext(timeoutCtx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	
	resp, err := fw.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}
	
	return nil
}


