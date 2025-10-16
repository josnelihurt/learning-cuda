package steps

import (
	"github.com/cucumber/godog"
)

func (tc *TestContext) theResponseShouldContainTransportFormat(expected string) error {
	return tc.ThenTheResponseShouldContainTransportFormat(expected)
}

func (tc *TestContext) theResponseShouldContainEndpoint(expected string) error {
	return tc.ThenTheResponseShouldContainEndpoint(expected)
}

func (tc *TestContext) fliptShouldHaveFlag(flagKey string) error {
	return tc.ThenFliptShouldHaveFlag(flagKey)
}

func (tc *TestContext) flagShouldBeEnabled(flagKey string) error {
	return tc.ThenFliptShouldHaveFlagWithValue(flagKey, true)
}

func (tc *TestContext) theResponseStatusShouldBe(statusCode int) error {
	return tc.ThenTheResponseStatusShouldBe(statusCode)
}

func (tc *TestContext) theResponseShouldContainStatus(status string) error {
	return tc.ThenTheResponseShouldContainHealthStatus(status)
}

func InitializeThenSteps(ctx *godog.ScenarioContext, tc *TestContext) {
	ctx.Step(`^the response should contain transport format "([^"]*)"$`, tc.theResponseShouldContainTransportFormat)
	ctx.Step(`^the response should contain endpoint "([^"]*)"$`, tc.theResponseShouldContainEndpoint)
	ctx.Step(`^Flipt should have flag "([^"]*)"$`, tc.fliptShouldHaveFlag)
	ctx.Step(`^flag "([^"]*)" should be enabled$`, tc.flagShouldBeEnabled)
	ctx.Step(`^the response status should be (\d+)$`, tc.theResponseStatusShouldBe)
	ctx.Step(`^the response should contain status "([^"]*)"$`, tc.theResponseShouldContainStatus)
}
