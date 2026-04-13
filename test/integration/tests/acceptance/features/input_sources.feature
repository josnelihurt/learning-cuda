Feature: Input Source Selection
    As a client application
    I want to list the available input sources

    Background:
        Given Flipt is running at "http://localhost:8081" with namespace "default"
        And the service is running at "https://localhost:8443"

    Scenario: List default input sources
        When I call ListInputs endpoint
        Then the response should succeed
        And the response should contain input source "lena" with type "static"
        And the response should contain input source "webcam" with type "camera"

    Scenario: Each input source has required fields
        When I call ListInputs endpoint
        Then the response should succeed
        And each input source should have a non-empty id
        And each input source should have a non-empty display name
        And each input source should have a valid type

    Scenario: At least one input source is marked as default
        When I call ListInputs endpoint
        Then the response should succeed
        And at least one input source should be marked as default