package acceptance

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestFeatureFlagsAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping acceptance test in short mode")
	}

	ctx := NewBDDContext(
		"http://localhost:8081",
		"default",
		"https://localhost:8443",
	)

	t.Run("GetStreamConfig returns default values when Flipt is clean", func(t *testing.T) {
		t.Log("Given: Flipt is clean and service is running")
		require.NoError(t, ctx.GivenFliptIsClean(), "Failed to clean Flipt")
		require.NoError(t, ctx.GivenTheServiceIsRunning(), "Service is not running")
		require.NoError(t, ctx.GivenConfigHasDefaultValues("json", "/ws"), "Failed to set default config")

		t.Log("When: I call GetStreamConfig")
		require.NoError(t, ctx.WhenICallGetStreamConfig(), "Failed to call GetStreamConfig")

		t.Log("Then: The response should contain default values")
		require.NoError(t, ctx.ThenTheResponseShouldContainTransportFormat("json"), "Transport format mismatch")
		require.NoError(t, ctx.ThenTheResponseShouldContainEndpoint("/ws"), "Endpoint mismatch")
	})

	t.Run("Sync creates flags in Flipt with correct values", func(t *testing.T) {
		t.Log("Given: Flipt is clean and service is running")
		require.NoError(t, ctx.GivenFliptIsClean(), "Failed to clean Flipt")
		require.NoError(t, ctx.GivenTheServiceIsRunning(), "Service is not running")

		t.Log("When: I call SyncFeatureFlags")
		require.NoError(t, ctx.WhenICallSyncFeatureFlags(), "Failed to call SyncFeatureFlags")
		require.NoError(t, ctx.WhenIWaitForFlagsToBeSynced(), "Failed to wait for sync")

		t.Log("Then: Flipt should have the synced flags")
		require.NoError(t, ctx.ThenFliptShouldHaveFlag("ws_transport_format"), "Flag not found")
		require.NoError(t, ctx.ThenFliptShouldHaveFlag("observability_enabled"), "Flag not found")
		require.NoError(t, ctx.ThenFliptShouldHaveFlagWithValue("observability_enabled", true), "Flag value mismatch")
	})

	t.Run("GetStreamConfig uses Flipt values when flags exist", func(t *testing.T) {
		t.Log("Given: Flags are already synced from previous test and service is running")
		require.NoError(t, ctx.GivenTheServiceIsRunning(), "Service is not running")

		t.Log("When: I call GetStreamConfig")
		require.NoError(t, ctx.WhenICallGetStreamConfig(), "Failed to call GetStreamConfig")

		t.Log("Then: The response should contain values from Flipt")
		require.NoError(t, ctx.ThenTheResponseShouldContainTransportFormat("json"), "Transport format mismatch")
		require.NoError(t, ctx.ThenTheResponseShouldContainEndpoint("/ws"), "Endpoint mismatch")
	})

	t.Run("GetStreamConfig falls back to defaults when flag evaluation fails", func(t *testing.T) {
		t.Log("Given: Flipt is clean (flags don't exist)")
		require.NoError(t, ctx.GivenFliptIsClean(), "Failed to clean Flipt")
		require.NoError(t, ctx.GivenTheServiceIsRunning(), "Service is not running")

		t.Log("When: I call GetStreamConfig")
		require.NoError(t, ctx.WhenICallGetStreamConfig(), "Failed to call GetStreamConfig")

		t.Log("Then: The response should contain fallback values")
		require.NoError(t, ctx.ThenTheResponseShouldContainTransportFormat("json"), "Transport format mismatch")
		require.NoError(t, ctx.ThenTheResponseShouldContainEndpoint("/ws"), "Endpoint mismatch")
	})
}
