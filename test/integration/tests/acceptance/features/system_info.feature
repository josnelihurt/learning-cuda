Feature: System Information
  As a client application
  I want to retrieve consolidated system information
  So that I can display version and environment information in the UI

  Background:
    Given Flipt is running at "http://localhost:8081" with namespace "default"
    And the service is running at "https://localhost:8443"

  Scenario: Get system info successfully
    When I call GetSystemInfo
    Then the response should be successful
    And the response should include version information

  Scenario: Response includes all version fields
    When I call GetSystemInfo
    Then the version should have go_version
    And the version should have cpp_version
    And the version should have proto_version
    And the version should have branch
    And the version should have build_time
    And the version should have commit_hash

  Scenario: Response includes environment
    When I call GetSystemInfo
    Then the response should include environment
    And the environment should be "development" or "production"

  Scenario: All fields are non-empty
    When I call GetSystemInfo
    Then the go_version should not be empty
    And the cpp_version should not be empty
    And the proto_version should not be empty
    And the branch should not be empty
    And the build_time should not be empty
    And the commit_hash should not be empty
    And the environment should not be empty
