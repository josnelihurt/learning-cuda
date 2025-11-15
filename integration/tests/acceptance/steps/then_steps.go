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

func (tc *TestContext) theFilterListShouldHaveAtLeastNFilters(count int) error {
	return tc.ThenTheFilterListShouldHaveAtLeastNFilters(count)
}

func (tc *TestContext) theFilterListShouldInclude(filterID string) error {
	return tc.ThenTheFilterListShouldInclude(filterID)
}

func (tc *TestContext) theGenericFilterShouldHaveParameter(filterID, paramID string) error {
	return tc.ThenTheGenericFilterShouldHaveParameter(filterID, paramID)
}

func (tc *TestContext) theGenericParameterShouldBeOfType(filterID, paramID, paramType string) error {
	return tc.ThenTheGenericParameterShouldBeOfType(filterID, paramID, paramType)
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

func (tc *TestContext) theResponseShouldContainVideoNamed(id string) error {
	return tc.ThenResponseShouldContainVideo(id)
}

func (tc *TestContext) responseShouldContainVideoNamed(id string) error {
	return tc.ThenResponseShouldContainVideo(id)
}

func (tc *TestContext) eachVideoShouldHaveANonEmptyID() error {
	return tc.eachVideoShouldHaveNonEmptyID()
}

func (tc *TestContext) eachVideoShouldHaveANonEmptyDisplayName() error {
	return tc.eachVideoShouldHaveNonEmptyDisplayName()
}

func (tc *TestContext) eachVideoShouldHaveANonEmptyPath() error {
	return tc.eachVideoShouldHaveNonEmptyPath()
}

func (tc *TestContext) eachVideoShouldHaveANonEmptyPreviewImagePath() error {
	return tc.eachVideoShouldHaveNonEmptyPreviewImagePath()
}

func (tc *TestContext) atLeastOneVideoShouldBeMarkedDefault() error {
	return tc.atLeastOneVideoShouldBeMarkedAsDefault()
}

func (tc *TestContext) theResponseShouldContainUploadedVideoDetails() error {
	return tc.ThenTheResponseShouldContainUploadedVideoDetails()
}

func (tc *TestContext) theResponseShouldContainPreviewImagePath() error {
	return tc.ThenTheResponseShouldContainPreviewImagePath()
}

func (tc *TestContext) thePreviewFileShouldExistOnFilesystem() error {
	return tc.ThenThePreviewFileShouldExistOnFilesystem()
}

func (tc *TestContext) thePreviewShouldBeAValidPNGImage() error {
	return tc.ThenThePreviewShouldBeAValidPNGImage()
}

func (tc *TestContext) theVideoShouldHavePreviewImagePath(videoID string) error {
	return tc.ThenTheVideoShouldHavePreviewImagePath(videoID)
}

func (tc *TestContext) theFrameShouldHaveField(fieldName string) error {
	return tc.ThenTheFrameShouldHaveField(fieldName)
}

func (tc *TestContext) theFrameIDShouldBe(expectedID int) error {
	return tc.ThenTheFrameIDShouldBe(expectedID)
}

func (tc *TestContext) allFrameIDsShouldBeSequentialStartingFrom(startID int) error {
	return tc.ThenAllFrameIDsShouldBeSequentialStartingFrom(startID)
}

func (tc *TestContext) frameIDShouldComeBeforeFrameID(firstID, secondID int) error {
	return tc.ThenFrameIDShouldComeBeforeFrameID(firstID, secondID)
}

func (tc *TestContext) frameIDShouldBeTheLastCollectedFrameID(frameID int) error {
	return tc.ThenFrameIDShouldBeTheLastCollectedFrameID(frameID)
}

func (tc *TestContext) eachFramesFrameIDShouldMatchItsFrameNumber() error {
	return tc.ThenEachFramesFrameIDShouldMatchItsFrameNumber()
}

func (tc *TestContext) theMetadataShouldContainFrames(frameCount int) error {
	return tc.ThenTheMetadataShouldContainFrames(frameCount)
}

func (tc *TestContext) frameShouldHaveASHA256Hash(frameID int) error {
	return tc.ThenFrameShouldHaveASHA256Hash(frameID)
}

func (tc *TestContext) iCanRetrieveMetadataForFrameID(frameID int) error {
	return tc.ThenICanRetrieveMetadataForFrameID(frameID)
}

func (tc *TestContext) theResponseShouldIncludeLogLevel(expectedLevel string) error {
	return tc.ThenTheResponseShouldIncludeLogLevel(expectedLevel)
}

func (tc *TestContext) theResponseShouldIncludeConsoleLoggingEnabled() error {
	return tc.ThenTheResponseShouldIncludeConsoleLoggingEnabled()
}

func (tc *TestContext) theResponseShouldIncludeConsoleLoggingDisabled() error {
	return tc.ThenTheResponseShouldIncludeConsoleLoggingDisabled()
}

func (tc *TestContext) theLogsShouldBeWrittenToBackendLogger() error {
	return tc.ThenTheLogsShouldBeWrittenToBackendLogger()
}

func (tc *TestContext) theResponseShouldReturnHTTP(code int) error {
	return tc.ThenTheResponseShouldReturnHTTP200()
}

func (tc *TestContext) fliptShouldStillHaveAllFlagsConfiguredCorrectly() error {
	// Verify all expected flags exist
	expectedFlags := []string{
		"ws_transport_format",
		"observability_enabled",
		"frontend_log_level",
		"frontend_console_logging",
	}

	for _, flagKey := range expectedFlags {
		if err := tc.ThenFliptShouldHaveFlag(flagKey); err != nil {
			return err
		}
	}
	return nil
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
	ctx.Step(`^the filter list should have at least (\d+) filter$`, tc.theFilterListShouldHaveAtLeastNFilters)
	ctx.Step(`^the filter list should include "([^"]*)"$`, tc.theFilterListShouldInclude)
	ctx.Step(`^the filter "([^"]*)" should be defined$`, tc.theFilterShouldBeDefined)
	ctx.Step(`^the filter "([^"]*)" should have parameter "([^"]*)"$`, tc.theFilterShouldHaveParameter)
	ctx.Step(`^the parameter "([^"]*)" should be of type "([^"]*)"$`, tc.theParameterShouldBeOfType)
	ctx.Step(`^the parameter "([^"]*)" should have at least (\d+) options$`, tc.theParameterShouldHaveAtLeastNOptions)
	ctx.Step(`^the generic filter "([^"]*)" should have parameter "([^"]*)"$`, tc.theGenericFilterShouldHaveParameter)
	ctx.Step(`^the generic parameter "([^"]*)" in filter "([^"]*)" should be of type "([^"]*)"$`, func(paramID, filterID, paramType string) error {
		return tc.theGenericParameterShouldBeOfType(filterID, paramID, paramType)
	})
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
	ctx.Step(`^the response should contain video "([^"]*)"$`, tc.theResponseShouldContainVideoNamed)
	ctx.Step(`^response should contain video "([^"]*)"$`, tc.responseShouldContainVideoNamed)
	ctx.Step(`^each video should have a non-empty id$`, tc.eachVideoShouldHaveANonEmptyID)
	ctx.Step(`^each video should have a non-empty display name$`, tc.eachVideoShouldHaveANonEmptyDisplayName)
	ctx.Step(`^each video should have a non-empty path$`, tc.eachVideoShouldHaveANonEmptyPath)
	ctx.Step(`^each video should have a non-empty preview image path$`, tc.eachVideoShouldHaveANonEmptyPreviewImagePath)
	ctx.Step(`^at least one video should be marked as default$`, tc.atLeastOneVideoShouldBeMarkedDefault)
	ctx.Step(`^the response should contain the uploaded video details$`, tc.theResponseShouldContainUploadedVideoDetails)
	ctx.Step(`^the response should contain input source "([^"]*)" with type "([^"]*)"$`, tc.theResponseShouldContainInputSourceWithType)
	ctx.Step(`^the response should contain preview image path$`, tc.theResponseShouldContainPreviewImagePath)
	ctx.Step(`^the preview file should exist on filesystem$`, tc.thePreviewFileShouldExistOnFilesystem)
	ctx.Step(`^the preview should be a valid PNG image$`, tc.thePreviewShouldBeAValidPNGImage)
	ctx.Step(`^the video "([^"]*)" should have a preview image path$`, tc.theVideoShouldHavePreviewImagePath)
	ctx.Step(`^the frame should have field "([^"]*)"$`, tc.theFrameShouldHaveField)
	ctx.Step(`^the frame_id should be (\d+)$`, tc.theFrameIDShouldBe)
	ctx.Step(`^all frame_ids should be sequential starting from (\d+)$`, tc.allFrameIDsShouldBeSequentialStartingFrom)
	ctx.Step(`^frame_id (\d+) should come before frame_id (\d+)$`, tc.frameIDShouldComeBeforeFrameID)
	ctx.Step(`^frame_id (\d+) should be the last collected frame_id$`, tc.frameIDShouldBeTheLastCollectedFrameID)
	ctx.Step(`^each frame's frame_id should match its frame_number$`, tc.eachFramesFrameIDShouldMatchItsFrameNumber)
	ctx.Step(`^the metadata should contain (\d+) frames$`, tc.theMetadataShouldContainFrames)
	ctx.Step(`^frame (\d+) should have a SHA256 hash$`, tc.frameShouldHaveASHA256Hash)
	ctx.Step(`^I can retrieve metadata for frame_id (\d+)$`, tc.iCanRetrieveMetadataForFrameID)
	ctx.Step(`^the response should include log level "([^"]*)"$`, tc.theResponseShouldIncludeLogLevel)
	ctx.Step(`^the response should include console logging enabled$`, tc.theResponseShouldIncludeConsoleLoggingEnabled)
	ctx.Step(`^the response should include console logging disabled$`, tc.theResponseShouldIncludeConsoleLoggingDisabled)
	ctx.Step(`^the logs should be written to backend logger$`, tc.theLogsShouldBeWrittenToBackendLogger)
	ctx.Step(`^the response should return HTTP (\d+)$`, tc.theResponseShouldReturnHTTP)
	ctx.Step(`^Flipt should still have all flags configured correctly$`, tc.fliptShouldStillHaveAllFlagsConfiguredCorrectly)
	ctx.Step(`^the response should be successful$`, tc.theResponseShouldBeSuccessful)
	ctx.Step(`^the response should include version information$`, tc.theResponseShouldIncludeVersionInformation)
	ctx.Step(`^the version should have ([^"]*)$`, tc.theVersionShouldHave)
	ctx.Step(`^the response should include environment$`, tc.theResponseShouldIncludeEnvironment)
	ctx.Step(`^the environment should be "([^"]*)" or "([^"]*)"$`, tc.theEnvironmentShouldBeOneOf)
	ctx.Step(`^the ([^"]*) should not be empty$`, tc.theFieldShouldNotBeEmpty)
}

func (tc *TestContext) theResponseShouldBeSuccessful() error {
	return tc.ThenTheResponseShouldBeSuccessful()
}

func (tc *TestContext) theResponseShouldIncludeVersionInformation() error {
	return tc.ThenTheResponseShouldIncludeVersionInformation()
}

func (tc *TestContext) theVersionShouldHave(field string) error {
	return tc.ThenTheVersionShouldHave(field)
}

func (tc *TestContext) theResponseShouldIncludeEnvironment() error {
	return tc.ThenTheResponseShouldIncludeEnvironment()
}

func (tc *TestContext) theEnvironmentShouldBeOneOf(env1, env2 string) error {
	return tc.ThenTheEnvironmentShouldBe(env1)
}

func (tc *TestContext) theFieldShouldNotBeEmpty(field string) error {
	return tc.ThenTheFieldShouldNotBeEmpty(field)
}
