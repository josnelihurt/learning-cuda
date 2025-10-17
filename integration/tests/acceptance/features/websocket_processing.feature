Feature: WebSocket Image Processing
  As a client application
  I want to process images through WebSocket
  So that I can have real-time image processing with different transport formats

  Background:
    Given Flipt is running at "http://localhost:8081" with namespace "default"
    And the service is running at "https://localhost:8443"
    And I have image "lena.png"

  Scenario: Process frame with JSON transport and no filter using GPU
    When I connect to WebSocket with transport format "json"
    And I send WebSocket frame with filter "FILTER_TYPE_NONE", accelerator "ACCELERATOR_TYPE_CUDA", grayscale type ""
    Then the WebSocket response should be success
    And I close the WebSocket connection

  Scenario: Process frame with JSON transport and grayscale BT709 using CPU
    When I connect to WebSocket with transport format "json"
    And I send WebSocket frame with filter "FILTER_TYPE_GRAYSCALE", accelerator "ACCELERATOR_TYPE_CPU", grayscale type "GRAYSCALE_TYPE_BT709"
    Then the WebSocket response should be success
    And I close the WebSocket connection

  Scenario: WebSocket frame with empty request should fail
    When I connect to WebSocket with transport format "json"
    And I send invalid WebSocket frame "empty_request"
    Then the WebSocket response should be error
    And I close the WebSocket connection

  Scenario: WebSocket frame with empty image should fail
    When I connect to WebSocket with transport format "json"
    And I send invalid WebSocket frame "empty_image"
    Then the WebSocket response should be error
    And I close the WebSocket connection

