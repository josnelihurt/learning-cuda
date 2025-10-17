Feature: Dynamic Tools Configuration
  As a client application
  I want to retrieve available tools from the backend
  So that I can display them dynamically based on environment

  Background:
    Given Flipt is running at "http://localhost:8081" with namespace "default"
    And the service is running at "https://localhost:8443"

  Scenario: Retrieve available tools grouped by category
    When I call GetAvailableTools
    Then the response should contain tool categories
    And the categories should include "Observability"
    And the categories should include "Features"
    And the categories should include "Testing"

  Scenario: Tools have required fields
    When I call GetAvailableTools
    Then each tool should have an "id"
    And each tool should have a "name"
    And each tool should have a "type"

  Scenario: URL tools have resolved URLs
    When I call GetAvailableTools
    Then tools with type "url" should have a "url" field
    And the url should not be empty

  Scenario: Action tools have action identifiers
    When I call GetAvailableTools
    Then tools with type "action" should have an "action" field
    And the action should match known actions

  Scenario: Development environment returns localhost URLs
    Given the environment is "development"
    When I call GetAvailableTools
    And I find the tool "jaeger"
    Then the tool url should contain "localhost:16686"

  Scenario: Icons use local paths
    When I call GetAvailableTools
    And I find any tool with an icon
    Then the icon_path should start with "/static/img/tools/"

