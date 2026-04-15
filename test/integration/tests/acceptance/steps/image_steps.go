package steps

import (
	"github.com/cucumber/godog"
)

func (tc *TestContext) iHaveImage(imageName string) error {
	return tc.GivenIHaveImage(imageName)
}

func (tc *TestContext) iCallProcessImageWith(filter, accelerator, grayscaleType string) error {
	return tc.WhenICallProcessImageWith(filter, accelerator, grayscaleType)
}

func (tc *TestContext) iCallProcessImageWithBlurFilter(accelerator string, kernelSize int, sigma float64, borderMode string, separableStr string) error {
	separable := separableStr == "true"
	return tc.WhenICallProcessImageWithBlurFilter(accelerator, kernelSize, sigma, borderMode, separable)
}

func (tc *TestContext) iCallProcessImageWithMultipleFilters(filters string, accelerator string, grayscaleType string, blurKernelSize int, blurSigma float64, blurBorderMode string, blurSeparableStr string) error {
	blurSeparable := blurSeparableStr == "true"
	return tc.WhenICallProcessImageWithMultipleFilters(filters, accelerator, grayscaleType, blurKernelSize, blurSigma, blurBorderMode, blurSeparable)
}

func (tc *TestContext) iCallProcessImageWithInvalidData(errorType string) error {
	return tc.WhenICallProcessImageWithInvalidData(errorType)
}

func (tc *TestContext) iCallStreamProcessVideo() error {
	return tc.WhenICallStreamProcessVideo()
}

func (tc *TestContext) theProcessingShouldSucceed() error {
	return tc.ThenTheProcessingShouldSucceed()
}

func (tc *TestContext) theProcessingShouldFail() error {
	return tc.ThenTheProcessingShouldFail()
}

func (tc *TestContext) theImageChecksumShouldMatch(imageName, filter, accelerator, grayscaleType string) error {
	return tc.ThenTheImageChecksumShouldMatch(imageName, filter, accelerator, grayscaleType)
}

func (tc *TestContext) theResponseShouldBeUnimplemented() error {
	return tc.ThenTheResponseShouldBeUnimplemented()
}

func InitializeImageSteps(ctx *godog.ScenarioContext, tc *TestContext) {
	ctx.Step(`^I have image "([^"]*)"$`, tc.iHaveImage)
	ctx.Step(`^I call ProcessImage with filter "([^"]*)", accelerator "([^"]*)", grayscale type "([^"]*)"$`, tc.iCallProcessImageWith)
	ctx.Step(`^I call ProcessImage with blur filter, accelerator "([^"]*)", kernel size (\d+), sigma ([\d.]+), border mode "([^"]*)", separable (true|false)$`, tc.iCallProcessImageWithBlurFilter)
	ctx.Step(`^I call ProcessImage with multiple filters "([^"]*)", accelerator "([^"]*)", grayscale type "([^"]*)", blur kernel size (\d+), blur sigma ([\d.]+), blur border mode "([^"]*)", blur separable (true|false)$`, tc.iCallProcessImageWithMultipleFilters)
	ctx.Step(`^I call ProcessImage with invalid data "([^"]*)"$`, tc.iCallProcessImageWithInvalidData)
	ctx.Step(`^I call StreamProcessVideo$`, tc.iCallStreamProcessVideo)
	ctx.Step(`^the processing should succeed$`, tc.theProcessingShouldSucceed)
	ctx.Step(`^the processing should fail$`, tc.theProcessingShouldFail)
	ctx.Step(`^the image checksum should match "([^"]*)" with filter "([^"]*)", accelerator "([^"]*)", grayscale "([^"]*)"$`, tc.theImageChecksumShouldMatch)
	ctx.Step(`^the response should be Unimplemented$`, tc.theResponseShouldBeUnimplemented)
}
