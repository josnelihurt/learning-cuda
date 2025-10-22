package acceptance_test

import (
	"context"
	"os"
	"testing"

	"github.com/cucumber/godog"
	"github.com/cucumber/godog/colors"
	"github.com/jrb/cuda-learning/integration/tests/acceptance/steps"
)

func TestFeatures(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping acceptance test in short mode")
	}

	resultsDir := ".ignore/test-results"
	if err := os.MkdirAll(resultsDir, 0o755); err != nil {
		t.Logf("Warning: failed to create test-results directory: %v", err)
	}

	suite := godog.TestSuite{
		ScenarioInitializer: InitializeScenario,
		Options: &godog.Options{
			Format:   "pretty,cucumber:" + resultsDir + "/cucumber-report.json",
			Paths:    []string{"features"},
			Output:   colors.Colored(os.Stdout),
			TestingT: t,
		},
	}

	if suite.Run() != 0 {
		t.Fatal("non-zero status returned, failed to run feature tests")
	}
}

func InitializeScenario(ctx *godog.ScenarioContext) {
	testCtx := steps.NewTestContext()

	ctx.Before(func(ctx context.Context, sc *godog.Scenario) (context.Context, error) {
		testCtx.Reset()
		return ctx, nil
	})

	ctx.After(func(ctx context.Context, sc *godog.Scenario, err error) (context.Context, error) {
		testCtx.CloseWebSocket()
		return ctx, nil
	})

	steps.InitializeGivenSteps(ctx, testCtx)
	steps.InitializeWhenSteps(ctx, testCtx)
	steps.InitializeThenSteps(ctx, testCtx)
	steps.InitializeImageSteps(ctx, testCtx)
	steps.InitializeInputSourceSteps(ctx, testCtx)
	steps.InitializeAvailableImagesSteps(ctx, testCtx)
}
