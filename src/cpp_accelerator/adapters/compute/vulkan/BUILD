load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("//bazel/defs:vulkan_glsl_to_spirv.bzl", "vulkan_kernel_embedded")

vulkan_kernel_embedded(
    name = "vk_grayscale_kernel",
    shader = "grayscale_kernel.comp",
)

vulkan_kernel_embedded(
    name = "vk_blur_kernel",
    shader = "blur_kernel.comp",
)

cc_library(
    name = "vulkan_context",
    srcs = ["vulkan_context.cpp"],
    hdrs = [
        "vulkan_context.h",
        "vulkan_compute_utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//src/cpp_accelerator/core:logger",
        "@spdlog",
        "@vulkan_hpp//:vulkan_hpp",
    ],
    linkopts = ["-lvulkan"],
)

cc_library(
    name = "vulkan_grayscale_filter",
    srcs = [
        "vulkan_grayscale_filter.cpp",
        ":vk_grayscale_kernel_spv_embed",
    ],
    hdrs = [
        "vulkan_grayscale_filter.h",
        ":vk_grayscale_kernel_blob_h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":vulkan_context",
        "//src/cpp_accelerator/core:logger",
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "@spdlog",
        "@vulkan_hpp//:vulkan_hpp",
    ],
    linkopts = ["-lvulkan"],
)

cc_library(
    name = "vulkan_blur_filter",
    srcs = [
        "vulkan_blur_filter.cpp",
        ":vk_blur_kernel_spv_embed",
    ],
    hdrs = [
        "vulkan_blur_filter.h",
        ":vk_blur_kernel_blob_h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":vulkan_context",
        "//src/cpp_accelerator/core:logger",
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "@spdlog",
        "@vulkan_hpp//:vulkan_hpp",
    ],
    linkopts = ["-lvulkan"],
)

cc_library(
    name = "vulkan_filter_factory",
    srcs = ["vulkan_filter_factory.cpp"],
    hdrs = ["vulkan_filter_factory.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":vulkan_blur_filter",
        ":vulkan_grayscale_filter",
        "//src/cpp_accelerator/application/engine:filter_interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "//proto:common_cc_proto",
    ],
    linkopts = ["-lvulkan"],
)

cc_test(
    name = "vulkan_grayscale_filter_test",
    srcs = ["vulkan_grayscale_filter_test.cpp"],
    data = [
        "//data:static_images/lena.png",
    ],
    deps = [
        ":vulkan_grayscale_filter",
        ":vulkan_context",
        "//src/cpp_accelerator/adapters/compute/cpu:grayscale_filter",
        "//src/cpp_accelerator/adapters/image_io:image_adapters",
        "//src/cpp_accelerator/domain/interfaces",
        "@googletest//:gtest_main",
    ],
    linkopts = ["-lvulkan"],
)

cc_test(
    name = "vulkan_blur_filter_test",
    srcs = ["vulkan_blur_filter_test.cpp"],
    data = [
        "//data:static_images/lena.png",
    ],
    deps = [
        ":vulkan_blur_filter",
        ":vulkan_context",
        "//src/cpp_accelerator/adapters/image_io:image_adapters",
        "//src/cpp_accelerator/domain/interfaces",
        "@googletest//:gtest_main",
    ],
    linkopts = ["-lvulkan"],
)
