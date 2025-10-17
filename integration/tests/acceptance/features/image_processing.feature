Feature: Image Processing via ConnectRPC
  As a client application
  I want to process images with different filters and accelerators
  So that I can transform images using GPU or CPU

  Background:
    Given Flipt is running at "http://localhost:8081" with namespace "default"
    And the service is running at "https://localhost:8443"
    And I have image "lena.png"

  Scenario: Process image with no filter using GPU
    When I call ProcessImage with filter "FILTER_TYPE_NONE", accelerator "ACCELERATOR_TYPE_CUDA", grayscale type ""
    Then the processing should succeed
    And the image checksum should match "lena.png" with filter "FILTER_TYPE_NONE", accelerator "ACCELERATOR_TYPE_CUDA", grayscale ""

  Scenario: Process image with no filter using CPU
    When I call ProcessImage with filter "FILTER_TYPE_NONE", accelerator "ACCELERATOR_TYPE_CPU", grayscale type ""
    Then the processing should succeed
    And the image checksum should match "lena.png" with filter "FILTER_TYPE_NONE", accelerator "ACCELERATOR_TYPE_CPU", grayscale ""

  Scenario: Process image with grayscale BT601 using GPU
    When I call ProcessImage with filter "FILTER_TYPE_GRAYSCALE", accelerator "ACCELERATOR_TYPE_CUDA", grayscale type "GRAYSCALE_TYPE_BT601"
    Then the processing should succeed
    And the image checksum should match "lena.png" with filter "FILTER_TYPE_GRAYSCALE", accelerator "ACCELERATOR_TYPE_CUDA", grayscale "GRAYSCALE_TYPE_BT601"

  Scenario: Process image with grayscale BT709 using GPU
    When I call ProcessImage with filter "FILTER_TYPE_GRAYSCALE", accelerator "ACCELERATOR_TYPE_CUDA", grayscale type "GRAYSCALE_TYPE_BT709"
    Then the processing should succeed
    And the image checksum should match "lena.png" with filter "FILTER_TYPE_GRAYSCALE", accelerator "ACCELERATOR_TYPE_CUDA", grayscale "GRAYSCALE_TYPE_BT709"

  Scenario: Process image with grayscale AVERAGE using GPU
    When I call ProcessImage with filter "FILTER_TYPE_GRAYSCALE", accelerator "ACCELERATOR_TYPE_CUDA", grayscale type "GRAYSCALE_TYPE_AVERAGE"
    Then the processing should succeed
    And the image checksum should match "lena.png" with filter "FILTER_TYPE_GRAYSCALE", accelerator "ACCELERATOR_TYPE_CUDA", grayscale "GRAYSCALE_TYPE_AVERAGE"

  Scenario: Process image with grayscale LIGHTNESS using GPU
    When I call ProcessImage with filter "FILTER_TYPE_GRAYSCALE", accelerator "ACCELERATOR_TYPE_CUDA", grayscale type "GRAYSCALE_TYPE_LIGHTNESS"
    Then the processing should succeed
    And the image checksum should match "lena.png" with filter "FILTER_TYPE_GRAYSCALE", accelerator "ACCELERATOR_TYPE_CUDA", grayscale "GRAYSCALE_TYPE_LIGHTNESS"

  Scenario: Process image with grayscale LUMINOSITY using GPU
    When I call ProcessImage with filter "FILTER_TYPE_GRAYSCALE", accelerator "ACCELERATOR_TYPE_CUDA", grayscale type "GRAYSCALE_TYPE_LUMINOSITY"
    Then the processing should succeed
    And the image checksum should match "lena.png" with filter "FILTER_TYPE_GRAYSCALE", accelerator "ACCELERATOR_TYPE_CUDA", grayscale "GRAYSCALE_TYPE_LUMINOSITY"

  Scenario: Process image with grayscale BT601 using CPU
    When I call ProcessImage with filter "FILTER_TYPE_GRAYSCALE", accelerator "ACCELERATOR_TYPE_CPU", grayscale type "GRAYSCALE_TYPE_BT601"
    Then the processing should succeed
    And the image checksum should match "lena.png" with filter "FILTER_TYPE_GRAYSCALE", accelerator "ACCELERATOR_TYPE_CPU", grayscale "GRAYSCALE_TYPE_BT601"

  Scenario: Process image with grayscale BT709 using CPU
    When I call ProcessImage with filter "FILTER_TYPE_GRAYSCALE", accelerator "ACCELERATOR_TYPE_CPU", grayscale type "GRAYSCALE_TYPE_BT709"
    Then the processing should succeed
    And the image checksum should match "lena.png" with filter "FILTER_TYPE_GRAYSCALE", accelerator "ACCELERATOR_TYPE_CPU", grayscale "GRAYSCALE_TYPE_BT709"

  Scenario: Process image with grayscale AVERAGE using CPU
    When I call ProcessImage with filter "FILTER_TYPE_GRAYSCALE", accelerator "ACCELERATOR_TYPE_CPU", grayscale type "GRAYSCALE_TYPE_AVERAGE"
    Then the processing should succeed
    And the image checksum should match "lena.png" with filter "FILTER_TYPE_GRAYSCALE", accelerator "ACCELERATOR_TYPE_CPU", grayscale "GRAYSCALE_TYPE_AVERAGE"

  Scenario: Process image with grayscale LIGHTNESS using CPU
    When I call ProcessImage with filter "FILTER_TYPE_GRAYSCALE", accelerator "ACCELERATOR_TYPE_CPU", grayscale type "GRAYSCALE_TYPE_LIGHTNESS"
    Then the processing should succeed
    And the image checksum should match "lena.png" with filter "FILTER_TYPE_GRAYSCALE", accelerator "ACCELERATOR_TYPE_CPU", grayscale "GRAYSCALE_TYPE_LIGHTNESS"

  Scenario: Process image with grayscale LUMINOSITY using CPU
    When I call ProcessImage with filter "FILTER_TYPE_GRAYSCALE", accelerator "ACCELERATOR_TYPE_CPU", grayscale type "GRAYSCALE_TYPE_LUMINOSITY"
    Then the processing should succeed
    And the image checksum should match "lena.png" with filter "FILTER_TYPE_GRAYSCALE", accelerator "ACCELERATOR_TYPE_CPU", grayscale "GRAYSCALE_TYPE_LUMINOSITY"

  Scenario: Fail to process empty image
    When I call ProcessImage with invalid data "empty_image"
    Then the processing should fail

  Scenario: Fail to process image with zero dimensions
    When I call ProcessImage with invalid data "zero_dimensions"
    Then the processing should fail


