package featureflags

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
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
	return grpcURL
}

func (fw *FliptWriter) SyncFlags(ctx context.Context, flags map[string]interface{}) error {
	log.Printf("Starting flag sync to Flipt at %s/api/v1/namespaces/%s/flags", fw.apiURL, fw.namespace)
	log.Printf("Flags to sync: %+v", flags)

	failedFlags := []string{}

	for key, value := range flags {
		switch v := value.(type) {
		case bool:
			log.Printf("Processing boolean flag '%s' with value %v", key, v)
			if err := fw.ensureBooleanFlagExists(ctx, key, v); err != nil {
				log.Printf("Failed to sync boolean flag '%s': %v", key, err)
				failedFlags = append(failedFlags, key)
			} else {
				log.Printf("Successfully synced boolean flag '%s'", key)
			}
		case string:
			if v != "" {
				log.Printf("Processing variant flag '%s' with value '%s'", key, v)
				if err := fw.ensureVariantFlagExists(ctx, key, v); err != nil {
					log.Printf("Failed to sync variant flag '%s': %v", key, err)
					failedFlags = append(failedFlags, key)
				} else {
					log.Printf("Successfully synced variant flag '%s'", key)
				}
			} else {
				log.Printf("Skipping empty string flag '%s'", key)
			}
		default:
			log.Printf("Skipping unsupported flag type '%s': %T", key, value)
		}
	}

	if len(failedFlags) > 0 {
		return fmt.Errorf("failed to sync %d flags: %v", len(failedFlags), failedFlags)
	}

	log.Printf("All flags synced successfully")
	return nil
}

func (fw *FliptWriter) ensureBooleanFlagExists(ctx context.Context, flagKey string, defaultEnabled bool) error {
	exists, err := fw.checkFlagExists(ctx, flagKey)
	if err != nil {
		return fmt.Errorf("failed to check flag existence: %w", err)
	}

	if exists {
		return nil
	}

	if err := fw.createBooleanFlag(ctx, flagKey, defaultEnabled); err != nil {
		return fmt.Errorf("failed to create boolean flag: %w", err)
	}

	if err := fw.createBooleanVariant(ctx, flagKey, defaultEnabled); err != nil {
		log.Printf("WARNING: Flag created but variant setup failed for '%s': %v", flagKey, err)
	}

	return nil
}

func (fw *FliptWriter) ensureVariantFlagExists(ctx context.Context, flagKey string, defaultValue string) error {
	exists, err := fw.checkFlagExists(ctx, flagKey)
	if err != nil {
		return fmt.Errorf("failed to check flag existence: %w", err)
	}

	if exists {
		return nil
	}

	if err := fw.createVariantFlag(ctx, flagKey); err != nil {
		return fmt.Errorf("failed to create variant flag: %w", err)
	}

	if err := fw.createStringVariant(ctx, flagKey, defaultValue); err != nil {
		log.Printf("WARNING: Flag created but variant setup failed for '%s': %v", flagKey, err)
	}

	return nil
}

func (fw *FliptWriter) checkFlagExists(ctx context.Context, flagKey string) (bool, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	url := fmt.Sprintf("%s/api/v1/namespaces/%s/flags/%s", fw.apiURL, fw.namespace, flagKey)
	log.Printf("Checking if flag exists: GET %s", url)

	req, err := http.NewRequestWithContext(timeoutCtx, "GET", url, nil)
	if err != nil {
		return false, err
	}

	resp, err := fw.client.Do(req)
	if err != nil {
		log.Printf("Error checking flag existence: %v", err)
		return false, err
	}
	defer resp.Body.Close()

	exists := resp.StatusCode == http.StatusOK
	log.Printf("Flag '%s' exists: %v (status: %d)", flagKey, exists, resp.StatusCode)
	return exists, nil
}

func (fw *FliptWriter) createBooleanFlag(ctx context.Context, flagKey string, defaultEnabled bool) error {
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

	log.Printf("Creating flag: POST %s with payload: %s", url, string(jsonData))

	req, err := http.NewRequestWithContext(timeoutCtx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := fw.client.Do(req)
	if err != nil {
		log.Printf("Error creating flag: %v", err)
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		bodyBytes, _ := io.ReadAll(resp.Body) //nolint:errcheck // Best effort error logging
		log.Printf("Failed to create flag '%s': status %d, body: %s", flagKey, resp.StatusCode, string(bodyBytes))
		return fmt.Errorf("unexpected status code: %d, body: %s", resp.StatusCode, string(bodyBytes))
	}

	log.Printf("Flag '%s' created successfully (status: %d)", flagKey, resp.StatusCode)
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

func (fw *FliptWriter) createVariantFlag(ctx context.Context, flagKey string) error {
	timeoutCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	url := fmt.Sprintf("%s/api/v1/namespaces/%s/flags", fw.apiURL, fw.namespace)

	payload := map[string]interface{}{
		"key":         flagKey,
		"name":        flagKey,
		"description": "Auto-synced from config.yaml",
		"enabled":     true,
		"type":        "VARIANT_FLAG_TYPE",
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	log.Printf("Creating variant flag: POST %s with payload: %s", url, string(jsonData))

	req, err := http.NewRequestWithContext(timeoutCtx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := fw.client.Do(req)
	if err != nil {
		log.Printf("Error creating variant flag: %v", err)
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		bodyBytes, _ := io.ReadAll(resp.Body) //nolint:errcheck // Best effort error logging
		log.Printf("Failed to create variant flag '%s': status %d, body: %s", flagKey, resp.StatusCode, string(bodyBytes))
		return fmt.Errorf("unexpected status code: %d, body: %s", resp.StatusCode, string(bodyBytes))
	}

	log.Printf("Variant flag '%s' created successfully (status: %d)", flagKey, resp.StatusCode)
	return nil
}

func (fw *FliptWriter) createStringVariant(ctx context.Context, flagKey string, value string) error {
	timeoutCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	url := fmt.Sprintf("%s/api/v1/namespaces/%s/flags/%s/variants", fw.apiURL, fw.namespace, flagKey)

	attachmentJSON, _ := json.Marshal(map[string]string{"value": value}) //nolint:errcheck // Best effort attachment

	payload := map[string]interface{}{
		"key":        value,
		"name":       value,
		"attachment": string(attachmentJSON),
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	log.Printf("Creating string variant: POST %s with payload: %s", url, string(jsonData))

	req, err := http.NewRequestWithContext(timeoutCtx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := fw.client.Do(req)
	if err != nil {
		log.Printf("Error creating string variant: %v", err)
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		bodyBytes, _ := io.ReadAll(resp.Body) //nolint:errcheck // Best effort error logging
		log.Printf("Failed to create string variant for '%s': status %d, body: %s", flagKey, resp.StatusCode, string(bodyBytes))
		return fmt.Errorf("unexpected status code: %d, body: %s", resp.StatusCode, string(bodyBytes))
	}

	log.Printf("String variant '%s' created successfully for flag '%s' (status: %d)", value, flagKey, resp.StatusCode)
	return nil
}

func (fw *FliptWriter) DeleteFlag(ctx context.Context, flagKey string) error {
	timeoutCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	url := fmt.Sprintf("%s/api/v1/namespaces/%s/flags/%s", fw.apiURL, fw.namespace, flagKey)
	log.Printf("Deleting flag: DELETE %s", url)

	req, err := http.NewRequestWithContext(timeoutCtx, "DELETE", url, nil)
	if err != nil {
		return err
	}

	resp, err := fw.client.Do(req)
	if err != nil {
		log.Printf("Error deleting flag: %v", err)
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		log.Printf("Flag '%s' not found (already deleted or never existed)", flagKey)
		return nil
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		bodyBytes, _ := io.ReadAll(resp.Body) //nolint:errcheck // Best effort error logging
		log.Printf("Failed to delete flag '%s': status %d, body: %s", flagKey, resp.StatusCode, string(bodyBytes))
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	log.Printf("Flag '%s' deleted successfully (status: %d)", flagKey, resp.StatusCode)
	return nil
}

func (fw *FliptWriter) DeleteAllFlags(ctx context.Context) error {
	timeoutCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	url := fmt.Sprintf("%s/api/v1/namespaces/%s/flags", fw.apiURL, fw.namespace)
	log.Printf("Listing all flags: GET %s", url)

	req, err := http.NewRequestWithContext(timeoutCtx, "GET", url, nil)
	if err != nil {
		return err
	}

	resp, err := fw.client.Do(req)
	if err != nil {
		log.Printf("Error listing flags: %v", err)
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body) //nolint:errcheck // Best effort error logging
		log.Printf("Failed to list flags: status %d, body: %s", resp.StatusCode, string(bodyBytes))
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	var result struct {
		Flags []struct {
			Key string `json:"key"`
		} `json:"flags"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return fmt.Errorf("failed to decode flags list: %w", err)
	}

	log.Printf("Found %d flags to delete", len(result.Flags))

	for _, flag := range result.Flags {
		if err := fw.DeleteFlag(ctx, flag.Key); err != nil {
			log.Printf("Warning: Failed to delete flag '%s': %v", flag.Key, err)
		}
	}

	log.Printf("All flags deleted successfully")
	return nil
}
