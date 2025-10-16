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

	validTypes := map[string]bool{"static": true, "camera": true}

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
