package steps

import (
	"context"
	"fmt"
	"time"

	"github.com/cucumber/godog"
)

func (tc *TestContext) theServerIsRunning() error {
	if tc.BDDContext == nil {
		return fmt.Errorf("BDD context not initialized - service URL not set")
	}
	return tc.GivenTheServiceIsRunning()
}

func (tc *TestContext) observabilityIsEnabled() error {
	if tc.BDDContext == nil {
		return fmt.Errorf("BDD context not initialized")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	flag, err := tc.BDDContext.GetFliptAPI().GetFlag(ctx, "observability_enabled")
	if err != nil {
		return tc.BDDContext.SetObservabilityFlag(ctx, true)
	}

	if flag == nil || !flag.Enabled {
		return tc.BDDContext.SetObservabilityFlag(ctx, true)
	}

	return nil
}

func (tc *TestContext) theFeatureFlagIsSetTo(flagKey, value string) error {
	if tc.BDDContext == nil {
		return fmt.Errorf("BDD context not initialized")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	exists, err := tc.BDDContext.FlagExists(ctx, flagKey)
	if err != nil {
		return fmt.Errorf("failed to check if flag exists: %w", err)
	}

	if exists {
		err = tc.BDDContext.GetFliptAPI().DeleteFlag(ctx, flagKey)
		if err != nil {
			return fmt.Errorf("failed to delete existing flag: %w", err)
		}
		time.Sleep(500 * time.Millisecond)
	}

	flags := map[string]interface{}{
		flagKey: value,
	}
	err = tc.BDDContext.GetFliptAPI().SyncFlags(ctx, flags)
	if err != nil {
		return fmt.Errorf("failed to set variant flag: %w", err)
	}

	time.Sleep(1 * time.Second)
	return nil
}

func (tc *TestContext) theFeatureFlagIsSetToTrue(flagKey string) error {
	return tc.theFeatureFlagIsSetToBoolean(flagKey, true)
}

func (tc *TestContext) theFeatureFlagIsSetToFalse(flagKey string) error {
	return tc.theFeatureFlagIsSetToBoolean(flagKey, false)
}

func (tc *TestContext) theFeatureFlagIsSetToBoolean(flagKey string, value bool) error {
	if tc.BDDContext == nil {
		return fmt.Errorf("BDD context not initialized")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	exists, err := tc.BDDContext.FlagExists(ctx, flagKey)
	if err != nil {
		return fmt.Errorf("failed to check if flag exists: %w", err)
	}

	if exists {
		err = tc.BDDContext.GetFliptAPI().DeleteFlag(ctx, flagKey)
		if err != nil {
			return fmt.Errorf("failed to delete existing flag: %w", err)
		}
		time.Sleep(500 * time.Millisecond)
	}

	flags := map[string]interface{}{
		flagKey: value,
	}
	err = tc.BDDContext.GetFliptAPI().SyncFlags(ctx, flags)
	if err != nil {
		return fmt.Errorf("failed to set boolean flag: %w", err)
	}

	time.Sleep(1 * time.Second)
	return nil
}

func (tc *TestContext) theFeatureFlagIsNotSet(flagKey string) error {
	if tc.BDDContext == nil {
		return fmt.Errorf("BDD context not initialized")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := tc.BDDContext.GetFliptAPI().DeleteFlag(ctx, flagKey)
	if err != nil {
		return fmt.Errorf("failed to delete flag: %w", err)
	}

	time.Sleep(500 * time.Millisecond)
	return nil
}

func (tc *TestContext) theFrontendIsConfigured() error {
	return nil
}

func InitializeFrontendLoggingSteps(ctx *godog.ScenarioContext, tc *TestContext) {
	ctx.Step(`^the server is running$`, tc.theServerIsRunning)
	ctx.Step(`^observability is enabled$`, tc.observabilityIsEnabled)
	ctx.Step(`^the feature flag "([^"]*)" is set to "([^"]*)"$`, tc.theFeatureFlagIsSetTo)
	ctx.Step(`^the feature flag "([^"]*)" is set to true$`, tc.theFeatureFlagIsSetToTrue)
	ctx.Step(`^the feature flag "([^"]*)" is set to false$`, tc.theFeatureFlagIsSetToFalse)
	ctx.Step(`^the feature flag "([^"]*)" is not set$`, tc.theFeatureFlagIsNotSet)
	ctx.Step(`^the frontend is configured$`, tc.theFrontendIsConfigured)
}

