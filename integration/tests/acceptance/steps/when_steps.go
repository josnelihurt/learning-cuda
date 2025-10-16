package steps

import (
	"github.com/cucumber/godog"
)

func (tc *TestContext) iCallTheGetStreamConfigEndpoint() error {
	return tc.WhenICallGetStreamConfig()
}

func (tc *TestContext) iCallTheSyncFeatureFlagsEndpoint() error {
	return tc.WhenICallSyncFeatureFlags()
}

func (tc *TestContext) iWaitForFlagsToBeSynchronized() error {
	return tc.WhenIWaitForFlagsToBeSynced()
}

func (tc *TestContext) iCallTheHealthEndpoint() error {
	return tc.WhenICallHealthEndpoint()
}

func InitializeWhenSteps(ctx *godog.ScenarioContext, tc *TestContext) {
	ctx.Step(`^I call the GetStreamConfig endpoint$`, tc.iCallTheGetStreamConfigEndpoint)
	ctx.Step(`^I call the SyncFeatureFlags endpoint$`, tc.iCallTheSyncFeatureFlagsEndpoint)
	ctx.Step(`^I wait for flags to be synchronized$`, tc.iWaitForFlagsToBeSynchronized)
	ctx.Step(`^I call the health endpoint$`, tc.iCallTheHealthEndpoint)
}
