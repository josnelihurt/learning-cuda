Feature: Processor Capabilities
  As a client application
  I want to query the processor library capabilities
  So that I can discover supported filters and their parameters dynamically

  Background:
    Given Flipt is running at "http://localhost:8081" with namespace "default"
    And the service is running at "https://localhost:8443"

  Scenario: Query processor capabilities
    When I call GetProcessorStatus
    Then the response should include capabilities
    And the capabilities should have API version "2.1.0"

  Scenario: Capabilities include filter definitions
    When I call GetProcessorStatus
    Then the capabilities should have at least 1 filter
    And the filter "grayscale" should be defined

  Scenario: Grayscale filter has algorithm parameter
    When I call GetProcessorStatus
    Then the filter "grayscale" should have parameter "algorithm"
    And the parameter "algorithm" should be of type "select"
    And the parameter "algorithm" should have at least 3 options

  Scenario: Filters specify supported accelerators
    When I call GetProcessorStatus
    Then the filter "grayscale" should support accelerator "CUDA"
    And the filter "grayscale" should support accelerator "CPU"

