load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "cuda_filter_factory",
    srcs = ["cuda_filter_factory.cpp"],
    hdrs = ["cuda_filter_factory.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/cpp_accelerator/adapters/compute/cuda/filters:blur_processor",
        "//src/cpp_accelerator/adapters/compute/cuda/filters:grayscale_filter",
        "//src/cpp_accelerator/application/engine:filter_interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "//proto:common_cc_proto",
    ],
)
