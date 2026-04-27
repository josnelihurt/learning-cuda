load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
    name = "buffer_pool",
    srcs = ["buffer_pool.cpp"],
    hdrs = ["buffer_pool.h"],
    visibility = ["//visibility:public"],
    deps = [],
)

cc_library(
    name = "filter_pipeline",
    srcs = ["filter_pipeline.cpp"],
    hdrs = ["filter_pipeline.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":buffer_pool",
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
    ],
)

cc_test(
    name = "filter_pipeline_test",
    srcs = ["filter_pipeline_test.cpp"],
    data = [
        "//data:static_images/lena.png",
    ],
    deps = [
        ":filter_pipeline",
        "//src/cpp_accelerator/adapters/compute/cpu:blur_filter",
        "//src/cpp_accelerator/adapters/compute/cpu:grayscale_filter",
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "//src/cpp_accelerator/adapters/image_io:image_adapters",
        "@googletest//:gtest_main",
    ],
)

