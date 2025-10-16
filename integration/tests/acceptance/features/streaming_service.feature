Feature: Streaming Video Service
  As a client application
  I want to verify the StreamProcessVideo endpoint
  So that I know which features are implemented

  Background:
    Given Flipt is running at "http://localhost:8081" with namespace "default"
    And the service is running at "https://localhost:8443"

  Scenario: StreamProcessVideo endpoint should return Unimplemented
    When I call StreamProcessVideo
    Then the response should be Unimplemented

