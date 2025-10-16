Feature: Feature Flag Management via Flipt
  As a system operator
  I want to manage feature flags dynamically
  So that I can control system behavior without deployments

  Background:
    Given Flipt is running at "http://localhost:8081" with namespace "default"
    And the service is running at "https://localhost:8443"

  Scenario: GetStreamConfig returns default values when Flipt is clean
    Given Flipt has no flags configured
    And default config has transport format "json" and endpoint "/ws"
    When I call the GetStreamConfig endpoint
    Then the response should contain transport format "json"
    And the response should contain endpoint "/ws"

  Scenario: Sync creates flags in Flipt with correct values
    Given Flipt has no flags configured
    When I call the SyncFeatureFlags endpoint
    And I wait for flags to be synchronized
    Then Flipt should have flag "ws_transport_format"
    And Flipt should have flag "observability_enabled"
    And flag "observability_enabled" should be enabled

  Scenario: GetStreamConfig uses Flipt values when flags exist
    Given flags are already synced to Flipt
    When I call the GetStreamConfig endpoint
    Then the response should contain transport format "json"
    And the response should contain endpoint "/ws"

  Scenario: GetStreamConfig falls back to defaults when flags don't exist
    Given Flipt has no flags configured
    When I call the GetStreamConfig endpoint
    Then the response should contain transport format "json"
    And the response should contain endpoint "/ws"

  Scenario: Health endpoint returns healthy status
    When I call the health endpoint
    Then the response status should be 200
    And the response should contain status "healthy"

