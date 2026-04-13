Feature: List Available Static Images
    As a client application
    I want to list the available static images for processing

    Background:
        Given Flipt is running at "http://localhost:8081" with namespace "default"
        And the service is running at "https://localhost:8443"

    Scenario: List available static images
        When I call ListAvailableImages endpoint
        Then the response should succeed
        And the response should contain image "lena"
        And the response should contain image "mandrill"
        And the response should contain image "peppers"
        And the response should contain image "barbara"
        And the response should contain image "cameraman"
        And the response should contain image "airplane"

    Scenario: Each image has required fields
        When I call ListAvailableImages endpoint
        Then the response should succeed
        And each image should have a non-empty id
        And each image should have a non-empty display name
        And each image should have a non-empty path

    Scenario: At least one image is marked as default
        When I call ListAvailableImages endpoint
        Then the response should succeed
        And at least one image should be marked as default

