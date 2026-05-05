load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "context",
    srcs = [
        "compute_utils.cpp",
        "context.cpp",
    ],
    hdrs = [
        "context.h",
        "compute_utils.h",
    ],
    visibility = ["//src/cpp_accelerator/adapters/compute/vulkan:__subpackages__"],
    deps = [
        "//src/cpp_accelerator/core:logger",
        "@spdlog",
        "@vulkan_hpp//:vulkan_hpp",
    ],
    linkopts = ["-lvulkan"],
)
