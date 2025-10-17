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

func (tc *TestContext) theResponseShouldSucceed() error {
	return tc.ThenTheResponseShouldSucceed()
}

func (tc *TestContext) theResponseShouldIncludeCapabilities() error {
	return tc.ThenTheResponseShouldIncludeCapabilities()
}

func (tc *TestContext) theCapabilitiesShouldHaveAPIVersion(version string) error {
	return tc.ThenTheCapabilitiesShouldHaveAPIVersion(version)
}

func (tc *TestContext) theCapabilitiesShouldHaveAtLeastNFilters(count int) error {
	return tc.ThenTheCapabilitiesShouldHaveAtLeastNFilters(count)
}

func (tc *TestContext) theFilterShouldBeDefined(filterId string) error {
	return tc.ThenTheFilterShouldBeDefined(filterId)
}

func (tc *TestContext) theFilterShouldHaveParameter(filterId, paramId string) error {
	return tc.ThenTheFilterShouldHaveParameter(filterId, paramId)
}

func (tc *TestContext) theParameterShouldBeOfType(paramId, paramType string) error {
	return tc.ThenTheParameterShouldBeOfType(paramId, paramType)
}

func (tc *TestContext) theParameterShouldHaveAtLeastNOptions(paramId string, count int) error {
	return tc.ThenTheParameterShouldHaveAtLeastNOptions(paramId, count)
}

func (tc *TestContext) theFilterShouldSupportAccelerator(filterId, accelerator string) error {
	return tc.ThenTheFilterShouldSupportAccelerator(filterId, accelerator)
}

func InitializeThenSteps(ctx *godog.ScenarioContext, tc *TestContext) {
	ctx.Step(`^the response should contain transport format "([^"]*)"$`, tc.theResponseShouldContainTransportFormat)
	ctx.Step(`^the response should contain endpoint "([^"]*)"$`, tc.theResponseShouldContainEndpoint)
	ctx.Step(`^Flipt should have flag "([^"]*)"$`, tc.fliptShouldHaveFlag)
	ctx.Step(`^flag "([^"]*)" should be enabled$`, tc.flagShouldBeEnabled)
	ctx.Step(`^the response status should be (\d+)$`, tc.theResponseStatusShouldBe)
	ctx.Step(`^the response should contain status "([^"]*)"$`, tc.theResponseShouldContainStatus)
	ctx.Step(`^the response should succeed$`, tc.theResponseShouldSucceed)
	ctx.Step(`^the response should include capabilities$`, tc.theResponseShouldIncludeCapabilities)
	ctx.Step(`^the capabilities should have API version "([^"]*)"$`, tc.theCapabilitiesShouldHaveAPIVersion)
	ctx.Step(`^the capabilities should have at least (\d+) filter$`, tc.theCapabilitiesShouldHaveAtLeastNFilters)
	ctx.Step(`^the filter "([^"]*)" should be defined$`, tc.theFilterShouldBeDefined)
	ctx.Step(`^the filter "([^"]*)" should have parameter "([^"]*)"$`, tc.theFilterShouldHaveParameter)
	ctx.Step(`^the parameter "([^"]*)" should be of type "([^"]*)"$`, tc.theParameterShouldBeOfType)
	ctx.Step(`^the parameter "([^"]*)" should have at least (\d+) options$`, tc.theParameterShouldHaveAtLeastNOptions)
	ctx.Step(`^the filter "([^"]*)" should support accelerator "([^"]*)"$`, tc.theFilterShouldSupportAccelerator)
}
