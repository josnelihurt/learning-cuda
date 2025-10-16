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

func (tc *TestContext) iCallProcessImageWithInvalidData(errorType string) error {
	return tc.WhenICallProcessImageWithInvalidData(errorType)
}

func (tc *TestContext) iCallStreamProcessVideo() error {
	return tc.WhenICallStreamProcessVideo()
}

func (tc *TestContext) iConnectToWebSocket(transportFormat string) error {
	return tc.WhenIConnectToWebSocket(transportFormat)
}

func (tc *TestContext) iSendWebSocketFrame(filter, accelerator, grayscaleType string) error {
	return tc.WhenISendWebSocketFrame(filter, accelerator, grayscaleType)
}

func (tc *TestContext) iSendInvalidWebSocketFrame(errorType string) error {
	return tc.WhenISendInvalidWebSocketFrame(errorType)
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

func (tc *TestContext) theWebSocketResponseShouldBeSuccess() error {
	return tc.ThenTheWebSocketResponseShouldBeSuccess()
}

func (tc *TestContext) theWebSocketResponseShouldBeError() error {
	return tc.ThenTheWebSocketResponseShouldBeError()
}

func (tc *TestContext) theResponseShouldBeUnimplemented() error {
	return tc.ThenTheResponseShouldBeUnimplemented()
}

func (tc *TestContext) closeWebSocketConnection() error {
	tc.CloseWebSocket()
	return nil
}

func InitializeImageSteps(ctx *godog.ScenarioContext, tc *TestContext) {
	ctx.Step(`^I have image "([^"]*)"$`, tc.iHaveImage)
	ctx.Step(`^I call ProcessImage with filter "([^"]*)", accelerator "([^"]*)", grayscale type "([^"]*)"$`, tc.iCallProcessImageWith)
	ctx.Step(`^I call ProcessImage with invalid data "([^"]*)"$`, tc.iCallProcessImageWithInvalidData)
	ctx.Step(`^I call StreamProcessVideo$`, tc.iCallStreamProcessVideo)
	ctx.Step(`^I connect to WebSocket with transport format "([^"]*)"$`, tc.iConnectToWebSocket)
	ctx.Step(`^I send WebSocket frame with filter "([^"]*)", accelerator "([^"]*)", grayscale type "([^"]*)"$`, tc.iSendWebSocketFrame)
	ctx.Step(`^I send invalid WebSocket frame "([^"]*)"$`, tc.iSendInvalidWebSocketFrame)
	ctx.Step(`^the processing should succeed$`, tc.theProcessingShouldSucceed)
	ctx.Step(`^the processing should fail$`, tc.theProcessingShouldFail)
	ctx.Step(`^the image checksum should match "([^"]*)" with filter "([^"]*)", accelerator "([^"]*)", grayscale "([^"]*)"$`, tc.theImageChecksumShouldMatch)
	ctx.Step(`^the WebSocket response should be success$`, tc.theWebSocketResponseShouldBeSuccess)
	ctx.Step(`^the WebSocket response should be error$`, tc.theWebSocketResponseShouldBeError)
	ctx.Step(`^the response should be Unimplemented$`, tc.theResponseShouldBeUnimplemented)
	ctx.Step(`^I close the WebSocket connection$`, tc.closeWebSocketConnection)
}
