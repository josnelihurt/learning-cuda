Feature: Video Playback from Files
    As a client application
    I want to play video files with filters applied frame-by-frame
    So that I can process video content using GPU/CPU acceleration

    Background:
        Given Flipt is running at "http://localhost:8081" with namespace "default"
        And the service is running at "https://localhost:8443"

    Scenario: List available videos
        When I call ListAvailableVideos endpoint
        Then the response should succeed
        And the response should contain video "sample"

    Scenario: Each video has required fields
        When I call ListAvailableVideos endpoint
        Then the response should succeed
        And each video should have a non-empty id
        And each video should have a non-empty display name
        And each video should have a non-empty path
        And each video should have a non-empty preview image path

    Scenario: At least one video can be marked as default
        When I call ListAvailableVideos endpoint
        Then the response should succeed
        And at least one video should be marked as default

    Scenario: Upload valid MP4 video
        When I upload a valid MP4 video named "test-video.mp4"
        Then the upload should succeed
        And the response should contain the uploaded video details

    Scenario: Upload video larger than 100MB
        When I upload an MP4 video larger than 100MB
        Then the upload should fail with "file too large" error

    Scenario: Upload non-MP4 file as video
        When I upload a non-MP4 file named "test.avi"
        Then the upload should fail with "invalid format" error

    Scenario: Uploaded video appears in available videos list
        When I upload a valid MP4 video named "uploaded-video.mp4"
        And I call ListAvailableVideos endpoint
        Then the response should contain video "uploaded-video"

    Scenario: Video source appears in input sources list
        When I call ListInputs endpoint
        Then the response should succeed
        And the response should contain input source "sample" with type "video"

    Scenario: Uploaded video generates preview thumbnail
        When I upload a valid MP4 video named "preview-test.mp4"
        Then the upload should succeed
        And the response should contain preview image path
        And the preview file should exist on filesystem
        And the preview should be a valid PNG image

