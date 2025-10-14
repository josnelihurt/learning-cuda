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

type FliptWriter struct {
	apiURL    string
	namespace string
	client    *http.Client
}

func NewFliptWriter(grpcURL, namespace string) *FliptWriter {
	restAPIURL := convertGRPCToRESTURL(grpcURL)
	
	return &FliptWriter{
		apiURL:    restAPIURL,
		namespace: namespace,
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

func convertGRPCToRESTURL(grpcURL string) string {
	restURL := strings.Replace(grpcURL, ":9000", ":8081", 1)
	return restURL
}

func (fw *FliptWriter) SyncFlags(flags map[string]bool) error {
	failedFlags := []string{}
	
	for key, enabled := range flags {
		if err := fw.ensureFlagExists(key, enabled); err != nil {
			log.Printf("Failed to sync flag '%s': %v", key, err)
			failedFlags = append(failedFlags, key)
		}
	}
	
	if len(failedFlags) > 0 {
		return fmt.Errorf("failed to sync %d flags: %v", len(failedFlags), failedFlags)
	}
	
	return nil
}

func (fw *FliptWriter) ensureFlagExists(flagKey string, defaultEnabled bool) error {
	exists, err := fw.checkFlagExists(flagKey)
	if err != nil {
		return fmt.Errorf("failed to check flag existence: %w", err)
	}
	
	if exists {
		return nil
	}
	
	if err := fw.createFlag(flagKey, defaultEnabled); err != nil {
		return fmt.Errorf("failed to create flag: %w", err)
	}
	
	if err := fw.createBooleanVariant(flagKey, defaultEnabled); err != nil {
		log.Printf("WARNING: Flag created but variant setup failed for '%s': %v", flagKey, err)
	}
	
	return nil
}

func (fw *FliptWriter) checkFlagExists(flagKey string) (bool, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	url := fmt.Sprintf("%s/api/v1/namespaces/%s/flags/%s", fw.apiURL, fw.namespace, flagKey)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
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

func (fw *FliptWriter) createFlag(flagKey string, defaultEnabled bool) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
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
	
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
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

func (fw *FliptWriter) createBooleanVariant(flagKey string, defaultValue bool) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
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
	
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
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


