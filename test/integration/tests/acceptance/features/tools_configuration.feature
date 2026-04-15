Feature: Dynamic Tools Configuration
  As a client application
  I want to retrieve available tools from the backend
  So that I can display them dynamically based on environment

  Background:
    Given the service is running at "https://localhost:8443"

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

  Scenario: Icons use local paths
    When I call GetAvailableTools
    And I find any tool with an icon
    Then the icon_path should start with "/static/img/tools/"
