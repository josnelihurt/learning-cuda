cc_library(
    name = "interfaces",
    hdrs = [
        "grayscale_algorithm.h",
        "i_pixel_getter.h",
        "image_buffer.h",
        "image_sink.h",
        "image_source.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "processors",
    hdrs = ["processors/i_image_processor.h"],
    visibility = ["//visibility:public"],
    deps = [":interfaces"],
)

cc_library(
    name = "i_yolo_detector",
    hdrs = ["i_yolo_detector.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "//src/cpp_accelerator/domain/models:detection",
    ],
)

