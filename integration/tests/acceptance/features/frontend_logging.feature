Feature: Frontend Logging System
  As a developer
  I want frontend logs to be controlled by feature flags
  So that I can configure log levels and console output dynamically

  Background:
    Given the server is running
    And observability is enabled

  Scenario: Frontend receives log level configuration from backend
    Given the feature flag "frontend_log_level" is set to "DEBUG"
    And the feature flag "frontend_console_logging" is set to true
    When the client requests stream configuration
    Then the response should include log level "DEBUG"
    And the response should include console logging enabled

  Scenario: Frontend receives default log level when not configured
    Given the feature flag "frontend_log_level" is not set
    When the client requests stream configuration
    Then the response should include log level "INFO"
    And the response should include console logging enabled

  Scenario: Backend receives OTLP logs from frontend
    Given the frontend is configured
    When the backend receives OTLP logs at "/api/logs"
    Then the logs should be written to backend logger
    And the response should return HTTP 200

  Scenario: Feature flag changes affect log level
    Given the feature flag "frontend_log_level" is set to "ERROR"
    When the client requests stream configuration
    Then the response should include log level "ERROR"

  Scenario: Console logging can be disabled via feature flag
    Given the feature flag "frontend_console_logging" is set to false
    When the client requests stream configuration
    Then the response should include console logging disabled

