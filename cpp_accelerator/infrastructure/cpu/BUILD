cc_library(
    name = "grayscale_processor",
    srcs = ["grayscale_processor.cpp"],
    hdrs = ["grayscale_processor.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//cpp_accelerator/core:telemetry",
        "//cpp_accelerator/domain/interfaces",
        "//cpp_accelerator/domain/interfaces:processors",
        "@spdlog",
    ],
)

cc_library(
    name = "blur_filter",
    srcs = ["blur_filter.cpp"],
    hdrs = ["blur_filter.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//cpp_accelerator/domain/interfaces",
        "//cpp_accelerator/domain/interfaces/filters:filters",
    ],
)
