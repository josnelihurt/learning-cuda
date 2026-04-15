package steps

import (
	"github.com/cucumber/godog"
)

func (tc *TestContext) theServiceIsRunningAt(url string) error {
	tc.serviceURL = url
	tc.Reset()
	return tc.GivenTheServiceIsRunning()
}

func (tc *TestContext) defaultConfigHasTransportFormatAndEndpoint(format, endpoint string) error {
	return tc.GivenConfigHasDefaultValues(format, endpoint)
}

func (tc *TestContext) theEnvironmentIs(environment string) error {
	return nil
}

func InitializeGivenSteps(ctx *godog.ScenarioContext, tc *TestContext) {
	ctx.Step(`^the service is running at "([^"]*)"$`, tc.theServiceIsRunningAt)
	ctx.Step(`^default config has transport format "([^"]*)" and endpoint "([^"]*)"$`, tc.defaultConfigHasTransportFormatAndEndpoint)
	ctx.Step(`^the environment is "([^"]*)"$`, tc.theEnvironmentIs)
}
