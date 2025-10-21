package steps

import (
	"github.com/cucumber/godog"
)

func (tc *TestContext) iCallTheGetStreamConfigEndpoint() error {
	return tc.WhenICallGetStreamConfig()
}

func (tc *TestContext) iCallTheSyncFeatureFlagsEndpoint() error {
	return tc.WhenICallSyncFeatureFlags()
}

func (tc *TestContext) iWaitForFlagsToBeSynchronized() error {
	return tc.WhenIWaitForFlagsToBeSynced()
}

func (tc *TestContext) iCallTheHealthEndpoint() error {
	return tc.WhenICallHealthEndpoint()
}

func (tc *TestContext) iCallGetProcessorStatus() error {
	return tc.WhenICallGetProcessorStatus()
}

func (tc *TestContext) iCallGetAvailableTools() error {
	return tc.WhenICallGetAvailableTools()
}

func (tc *TestContext) iFindTheTool(toolID string) error {
	return tc.WhenIFindTheTool(toolID)
}

func (tc *TestContext) iFindAnyToolWithAnIcon() error {
	return tc.WhenIFindAnyToolWithAnIcon()
}

func (tc *TestContext) iUploadValidPNGImage(filename string) error {
	return tc.WhenIUploadValidPNGImage(filename)
}

func (tc *TestContext) iUploadLargePNGImage() error {
	return tc.WhenIUploadLargePNGImage()
}

func (tc *TestContext) iUploadNonPNGFile(filename string) error {
	return tc.WhenIUploadNonPNGFile(filename)
}

func (tc *TestContext) iCallListAvailableVideosEndpoint() error {
	return tc.WhenICallListAvailableVideos()
}

func (tc *TestContext) iCallListAvailableVideos() error {
	return tc.WhenICallListAvailableVideos()
}

func (tc *TestContext) iUploadValidMP4VideoNamed(filename string) error {
	return tc.WhenIUploadValidMP4Video(filename)
}

func (tc *TestContext) iUploadMP4VideoLargerThan100MB() error {
	return tc.WhenIUploadLargeMP4Video()
}

func (tc *TestContext) iUploadNonMP4FileNamed(filename string) error {
	return tc.WhenIUploadNonMP4File(filename)
}

func (tc *TestContext) iReceiveTheFirstVideoFrame() error {
	return tc.WhenIReceiveTheFirstVideoFrame()
}

func (tc *TestContext) iCollectVideoFrames(frameCount int) error {
	return tc.WhenICollectVideoFrames(frameCount)
}

func (tc *TestContext) iReceiveVideoFrames(frameCount int) error {
	return tc.WhenIReceiveVideoFrames(frameCount)
}

func (tc *TestContext) iReceiveTheFirstVideoFrames(frameCount int) error {
	return tc.WhenIReceiveVideoFrames(frameCount)
}

func (tc *TestContext) iQueryVideoMetadataFor(videoID string) error {
	return tc.WhenIQueryVideoMetadataFor(videoID)
}

func (tc *TestContext) theClientRequestsStreamConfiguration() error {
	return tc.WhenTheClientRequestsStreamConfiguration()
}

func (tc *TestContext) theBackendReceivesOTLPLogsAt(endpoint string) error {
	return tc.WhenTheBackendReceivesOTLPLogsAt(endpoint)
}

func InitializeWhenSteps(ctx *godog.ScenarioContext, tc *TestContext) {
	ctx.Step(`^I call the GetStreamConfig endpoint$`, tc.iCallTheGetStreamConfigEndpoint)
	ctx.Step(`^I call the SyncFeatureFlags endpoint$`, tc.iCallTheSyncFeatureFlagsEndpoint)
	ctx.Step(`^I wait for flags to be synchronized$`, tc.iWaitForFlagsToBeSynchronized)
	ctx.Step(`^I call the health endpoint$`, tc.iCallTheHealthEndpoint)
	ctx.Step(`^I call GetProcessorStatus$`, tc.iCallGetProcessorStatus)
	ctx.Step(`^I call GetAvailableTools$`, tc.iCallGetAvailableTools)
	ctx.Step(`^I find the tool "([^"]*)"$`, tc.iFindTheTool)
	ctx.Step(`^I find any tool with an icon$`, tc.iFindAnyToolWithAnIcon)
	ctx.Step(`^I upload a valid PNG image named "([^"]*)"$`, tc.iUploadValidPNGImage)
	ctx.Step(`^I upload a PNG image larger than 10MB$`, tc.iUploadLargePNGImage)
	ctx.Step(`^I upload a non-PNG file named "([^"]*)"$`, tc.iUploadNonPNGFile)
	ctx.Step(`^I call ListAvailableVideos endpoint$`, tc.iCallListAvailableVideosEndpoint)
	ctx.Step(`^I call ListAvailableVideos$`, tc.iCallListAvailableVideos)
	ctx.Step(`^I upload a valid MP4 video named "([^"]*)"$`, tc.iUploadValidMP4VideoNamed)
	ctx.Step(`^I upload an MP4 video larger than 100MB$`, tc.iUploadMP4VideoLargerThan100MB)
	ctx.Step(`^I upload a non-MP4 file named "([^"]*)"$`, tc.iUploadNonMP4FileNamed)
	ctx.Step(`^I receive the first video frame$`, tc.iReceiveTheFirstVideoFrame)
	ctx.Step(`^I collect (\d+) video frames$`, tc.iCollectVideoFrames)
	ctx.Step(`^I receive (\d+) video frames$`, tc.iReceiveVideoFrames)
	ctx.Step(`^I receive the first (\d+) video frames$`, tc.iReceiveTheFirstVideoFrames)
	ctx.Step(`^I query video metadata for "([^"]*)"$`, tc.iQueryVideoMetadataFor)
	ctx.Step(`^the client requests stream configuration$`, tc.theClientRequestsStreamConfiguration)
	ctx.Step(`^the backend receives OTLP logs at "([^"]*)"$`, tc.theBackendReceivesOTLPLogsAt)
}
