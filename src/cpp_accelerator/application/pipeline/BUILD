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
        "//cpp_accelerator/domain/interfaces",
        "//cpp_accelerator/domain/interfaces/filters:filters",
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
        "//cpp_accelerator/infrastructure/cpu:blur_filter",
        "//cpp_accelerator/infrastructure/cpu:grayscale_filter",
        "//cpp_accelerator/domain/interfaces",
        "//cpp_accelerator/domain/interfaces/filters:filters",
        "//cpp_accelerator/infrastructure/image:image_adapters",
        "@googletest//:gtest_main",
    ],
)

