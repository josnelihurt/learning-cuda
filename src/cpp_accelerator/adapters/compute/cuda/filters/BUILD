load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
    name = "blur_processor",
    srcs = ["blur_processor_wrapper.cpp"],
    hdrs = ["blur_processor.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/cpp_accelerator/adapters/compute/cuda/kernels:kernels",
        "//src/cpp_accelerator/core:telemetry",
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces/processors",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "@spdlog",
    ],
)

cc_library(
    name = "cuda_wrappers",
    hdrs = ["blur_processor.h"],
    visibility = ["//visibility:public"],
    deps = [":blur_processor"],
)

cc_library(
    name = "grayscale_filter",
    srcs = ["grayscale_filter.cpp"],
    hdrs = ["grayscale_filter.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/cpp_accelerator/adapters/compute/cuda/kernels:kernels",
        "//src/cpp_accelerator/core:telemetry",
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "@spdlog",
    ],
)

cc_test(
    name = "blur_processor_test",
    srcs = ["blur_processor_test.cpp"],
    data = ["//data:static_images/lena.png"],
    deps = [
        ":blur_processor",
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "//src/cpp_accelerator/adapters/image_io:image_adapters",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "grayscale_filter_test",
    srcs = ["grayscale_filter_test.cpp"],
    data = ["//data:static_images/lena.png"],
    deps = [
        ":grayscale_filter",
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "//src/cpp_accelerator/adapters/compute/cpu:grayscale_filter",
        "//src/cpp_accelerator/adapters/image_io:image_adapters",
        "@googletest//:gtest_main",
    ],
)
