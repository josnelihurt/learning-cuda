package steps

import (
	"fmt"

	"github.com/cucumber/godog"
)

func InitializeInputSourceSteps(ctx *godog.ScenarioContext, tc *TestContext) {
	ctx.Step(`^I call ListInputs endpoint$`, tc.iCallListInputsEndpoint)
	ctx.Step(`^the response should contain input source "([^"]*)" with type "([^"]*)"$`, tc.theResponseShouldContainInputSourceWithType)
	ctx.Step(`^each input source should have a non-empty id$`, tc.eachInputSourceShouldHaveNonEmptyID)
	ctx.Step(`^each input source should have a non-empty display name$`, tc.eachInputSourceShouldHaveNonEmptyDisplayName)
	ctx.Step(`^each input source should have a valid type$`, tc.eachInputSourceShouldHaveValidType)
	ctx.Step(`^at least one input source should be marked as default$`, tc.atLeastOneInputSourceShouldBeDefault)
}

func (tc *TestContext) iCallListInputsEndpoint() error {
	return tc.WhenICallListInputs()
}

func (tc *TestContext) theResponseShouldContainInputSourceWithType(id, sourceType string) error {
	return tc.ThenResponseShouldContainInputSource(id, sourceType)
}

func (tc *TestContext) eachInputSourceShouldHaveNonEmptyID() error {
	sources, err := tc.GetInputSourcesFromResponse()
	if err != nil {
		return err
	}

	for _, src := range sources {
		if src.Id == "" {
			return fmt.Errorf("found input source with empty id")
		}
	}
	return nil
}

func (tc *TestContext) eachInputSourceShouldHaveNonEmptyDisplayName() error {
	sources, err := tc.GetInputSourcesFromResponse()
	if err != nil {
		return err
	}

	for _, src := range sources {
		if src.DisplayName == "" {
			return fmt.Errorf("input source %s has empty display name", src.Id)
		}
	}
	return nil
}

func (tc *TestContext) eachInputSourceShouldHaveValidType() error {
	sources, err := tc.GetInputSourcesFromResponse()
	if err != nil {
		return err
	}

	validTypes := map[string]bool{"static": true, "camera": true, "video": true}

	for _, src := range sources {
		if !validTypes[src.Type] {
			return fmt.Errorf("input source %s has invalid type: %s", src.Id, src.Type)
		}
	}
	return nil
}

func (tc *TestContext) atLeastOneInputSourceShouldBeDefault() error {
	sources, err := tc.GetInputSourcesFromResponse()
	if err != nil {
		return err
	}

	for _, src := range sources {
		if src.IsDefault {
			return nil
		}
	}

	return fmt.Errorf("no input source is marked as default")
}

func InitializeAvailableImagesSteps(ctx *godog.ScenarioContext, tc *TestContext) {
	ctx.Step(`^I call ListAvailableImages endpoint$`, tc.iCallListAvailableImagesEndpoint)
	ctx.Step(`^the response should contain image "([^"]*)"$`, tc.theResponseShouldContainImage)
	ctx.Step(`^each image should have a non-empty id$`, tc.eachImageShouldHaveNonEmptyID)
	ctx.Step(`^each image should have a non-empty display name$`, tc.eachImageShouldHaveNonEmptyDisplayName)
	ctx.Step(`^each image should have a non-empty path$`, tc.eachImageShouldHaveNonEmptyPath)
	ctx.Step(`^at least one image should be marked as default$`, tc.atLeastOneImageShouldBeDefault)
}

func (tc *TestContext) iCallListAvailableImagesEndpoint() error {
	return tc.WhenICallListAvailableImages()
}

func (tc *TestContext) theResponseShouldContainImage(id string) error {
	return tc.ThenResponseShouldContainImage(id)
}

func (tc *TestContext) eachImageShouldHaveNonEmptyID() error {
	images, err := tc.GetImagesFromResponse()
	if err != nil {
		return err
	}

	for _, img := range images {
		if img.Id == "" {
			return fmt.Errorf("found image with empty id")
		}
	}
	return nil
}

func (tc *TestContext) eachImageShouldHaveNonEmptyDisplayName() error {
	images, err := tc.GetImagesFromResponse()
	if err != nil {
		return err
	}

	for _, img := range images {
		if img.DisplayName == "" {
			return fmt.Errorf("image %s has empty display name", img.Id)
		}
	}
	return nil
}

func (tc *TestContext) eachImageShouldHaveNonEmptyPath() error {
	images, err := tc.GetImagesFromResponse()
	if err != nil {
		return err
	}

	for _, img := range images {
		if img.Path == "" {
			return fmt.Errorf("image %s has empty path", img.Id)
		}
	}
	return nil
}

func (tc *TestContext) atLeastOneImageShouldBeDefault() error {
	images, err := tc.GetImagesFromResponse()
	if err != nil {
		return err
	}

	for _, img := range images {
		if img.IsDefault {
			return nil
		}
	}

	return fmt.Errorf("no image is marked as default")
}
