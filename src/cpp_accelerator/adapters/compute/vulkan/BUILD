load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "vulkan_filter_factory",
    srcs = ["vulkan_filter_factory.cpp"],
    hdrs = ["vulkan_filter_factory.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/cpp_accelerator/adapters/compute/vulkan/filters:blur_filter",
        "//src/cpp_accelerator/adapters/compute/vulkan/filters:grayscale_filter",
        "//src/cpp_accelerator/application/engine:filter_interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "//proto:common_cc_proto",
    ],
    linkopts = ["-lvulkan"],
)
