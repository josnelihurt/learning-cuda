package steps

import (
	"github.com/cucumber/godog"
)

func (tc *TestContext) fliptIsRunningAt(url, namespace string) error {
	tc.fliptURL = url
	tc.namespace = namespace
	tc.Reset()
	return nil
}

func (tc *TestContext) theServiceIsRunningAt(url string) error {
	tc.serviceURL = url
	tc.Reset()
	return tc.GivenTheServiceIsRunning()
}

func (tc *TestContext) fliptHasNoFlagsConfigured() error {
	return tc.GivenFliptIsClean()
}

func (tc *TestContext) defaultConfigHasTransportFormatAndEndpoint(format, endpoint string) error {
	return tc.GivenConfigHasDefaultValues(format, endpoint)
}

func (tc *TestContext) flagsAreAlreadySyncedToFlipt() error {
	if err := tc.WhenICallSyncFeatureFlags(); err != nil {
		return err
	}
	return tc.WhenIWaitForFlagsToBeSynced()
}

func (tc *TestContext) theEnvironmentIs(environment string) error {
	return nil
}

func (tc *TestContext) iStartVideoPlaybackForVideoWithDefaultFilters(videoID string) error {
	return tc.GivenIStartVideoPlaybackForVideoWithDefaultFilters(videoID)
}

func InitializeGivenSteps(ctx *godog.ScenarioContext, tc *TestContext) {
	ctx.Step(`^Flipt is running at "([^"]*)" with namespace "([^"]*)"$`, tc.fliptIsRunningAt)
	ctx.Step(`^the service is running at "([^"]*)"$`, tc.theServiceIsRunningAt)
	ctx.Step(`^Flipt has no flags configured$`, tc.fliptHasNoFlagsConfigured)
	ctx.Step(`^default config has transport format "([^"]*)" and endpoint "([^"]*)"$`, tc.defaultConfigHasTransportFormatAndEndpoint)
	ctx.Step(`^flags are already synced to Flipt$`, tc.flagsAreAlreadySyncedToFlipt)
	ctx.Step(`^the environment is "([^"]*)"$`, tc.theEnvironmentIs)
	ctx.Step(`^I start video playback for "([^"]*)" with default filters$`, tc.iStartVideoPlaybackForVideoWithDefaultFilters)
}
