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

func (tc *TestContext) theFilterShouldBeDefined(filterID string) error {
	return tc.ThenTheFilterShouldBeDefined(filterID)
}

func (tc *TestContext) theFilterShouldHaveParameter(filterID, paramID string) error {
	return tc.ThenTheFilterShouldHaveParameter(filterID, paramID)
}

func (tc *TestContext) theParameterShouldBeOfType(paramID, paramType string) error {
	return tc.ThenTheParameterShouldBeOfType(paramID, paramType)
}

func (tc *TestContext) theParameterShouldHaveAtLeastNOptions(paramID string, count int) error {
	return tc.ThenTheParameterShouldHaveAtLeastNOptions(paramID, count)
}

func (tc *TestContext) theFilterShouldSupportAccelerator(filterID, accelerator string) error {
	return tc.ThenTheFilterShouldSupportAccelerator(filterID, accelerator)
}

func (tc *TestContext) theResponseShouldContainToolCategories() error {
	return tc.ThenTheResponseShouldContainToolCategories()
}

func (tc *TestContext) theCategoriesShouldInclude(categoryName string) error {
	return tc.ThenTheCategoriesShouldInclude(categoryName)
}

func (tc *TestContext) eachToolShouldHaveField(fieldName string) error {
	return tc.ThenEachToolShouldHaveField(fieldName)
}

func (tc *TestContext) toolsWithTypeShouldHaveField(toolType, fieldName string) error {
	return tc.ThenToolsWithTypeShouldHaveField(toolType, fieldName)
}

func (tc *TestContext) theURLShouldNotBeEmpty() error {
	return tc.ThenTheURLShouldNotBeEmpty()
}

func (tc *TestContext) theActionShouldMatchKnownActions() error {
	return tc.ThenTheActionShouldMatchKnownActions()
}

func (tc *TestContext) theToolURLShouldContain(substring string) error {
	return tc.ThenTheToolURLShouldContain(substring)
}

func (tc *TestContext) theIconPathShouldStartWith(prefix string) error {
	return tc.ThenTheIconPathShouldStartWith(prefix)
}

func (tc *TestContext) theUploadShouldSucceed() error {
	return tc.ThenTheUploadShouldSucceed()
}

func (tc *TestContext) theUploadShouldFailWithError(expectedError string) error {
	return tc.ThenTheUploadShouldFailWithError(expectedError)
}

func (tc *TestContext) theResponseShouldContainUploadedImageDetails() error {
	return tc.ThenTheResponseShouldContainUploadedImageDetails()
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
	ctx.Step(`^the response should contain tool categories$`, tc.theResponseShouldContainToolCategories)
	ctx.Step(`^the categories should include "([^"]*)"$`, tc.theCategoriesShouldInclude)
	ctx.Step(`^each tool should have an? "([^"]*)"$`, tc.eachToolShouldHaveField)
	ctx.Step(`^tools with type "([^"]*)" should have an? "([^"]*)" field$`, tc.toolsWithTypeShouldHaveField)
	ctx.Step(`^the url should not be empty$`, tc.theURLShouldNotBeEmpty)
	ctx.Step(`^the action should match known actions$`, tc.theActionShouldMatchKnownActions)
	ctx.Step(`^the tool url should contain "([^"]*)"$`, tc.theToolURLShouldContain)
	ctx.Step(`^the icon_path should start with "([^"]*)"$`, tc.theIconPathShouldStartWith)
	ctx.Step(`^the upload should succeed$`, tc.theUploadShouldSucceed)
	ctx.Step(`^the upload should fail with "([^"]*)" error$`, tc.theUploadShouldFailWithError)
	ctx.Step(`^the response should contain the uploaded image details$`, tc.theResponseShouldContainUploadedImageDetails)
}
