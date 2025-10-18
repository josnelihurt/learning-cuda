Feature: Upload Images
    As a client application
    I want to upload custom PNG images
    So that I can use them for processing

    Background:
        Given Flipt is running at "http://localhost:8081" with namespace "default"
        And the service is running at "https://localhost:8443"

    Scenario: Upload valid PNG image
        When I upload a valid PNG image named "test-upload.png"
        Then the upload should succeed
        And the response should contain the uploaded image details

    Scenario: Upload image larger than 10MB
        When I upload a PNG image larger than 10MB
        Then the upload should fail with "file too large" error

    Scenario: Upload non-PNG file
        When I upload a non-PNG file named "test.jpg"
        Then the upload should fail with "invalid format" error

    Scenario: Uploaded image appears in available images list
        When I upload a valid PNG image named "uploaded-test.png"
        And I call ListAvailableImages endpoint
        Then the response should contain image "uploaded-test"

