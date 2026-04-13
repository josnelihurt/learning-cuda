Feature: Video Frame ID Tracking
  As a client application
  I want to receive sequential frame IDs for each video frame
  So that I can deterministically validate video playback and test with pre-calculated metadata

  Background:
    Given Flipt is running at "http://localhost:8081" with namespace "default"
    And the service is running at "https://localhost:8443"

  Scenario: E2E test video is available in video list
    When I call ListAvailableVideos
    Then response should contain video "e2e-test"
    And the video "e2e-test" should have a preview image path

  Scenario: Video frames include frame_id starting at zero
    Given I start video playback for "e2e-test" with default filters
    When I receive the first video frame
    Then the frame should have field "frame_id"
    And the frame_id should be 0

  Scenario: Frame IDs are sequential during playback
    Given I start video playback for "e2e-test" with default filters
    When I collect 50 video frames
    Then all frame_ids should be sequential starting from 0
    And frame_id 0 should come before frame_id 1
    And frame_id 49 should be the last collected frame_id

  Scenario: Frame ID matches frame number
    Given I start video playback for "e2e-test" with default filters
    When I receive 10 video frames
    Then each frame's frame_id should match its frame_number

  Scenario: E2E test video has 200 frames of metadata
    When I query video metadata for "e2e-test"
    Then the metadata should contain 200 frames
    And frame 0 should have a SHA256 hash
    And frame 199 should have a SHA256 hash

  Scenario: Frame metadata can validate frame hashes
    Given I start video playback for "e2e-test" with default filters
    When I receive the first 3 video frames
    Then I can retrieve metadata for frame_id 0
    And I can retrieve metadata for frame_id 1
    And I can retrieve metadata for frame_id 2

