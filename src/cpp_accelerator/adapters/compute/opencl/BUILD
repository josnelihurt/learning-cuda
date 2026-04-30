load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("//bazel/defs:opencl_cl_to_spirv.bzl", "opencl_kernel_embedded")

opencl_kernel_embedded(
    name = "grayscale_kernel",
    cl_src = "grayscale_kernel.cl",
)

opencl_kernel_embedded(
    name = "blur_kernel",
    cl_src = "blur_kernel.cl",
)

cc_library(
    name = "opencl_context",
    srcs = ["opencl_context.cpp"],
    hdrs = ["opencl_context.h"],
    defines = ["CL_TARGET_OPENCL_VERSION=300"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/cpp_accelerator/core:logger",
        "@opencl_sdk//:opencl",
        "@spdlog",
    ],
    linkopts = ["-l:libOpenCL.so.1"],
)

cc_library(
    name = "opencl_grayscale_filter",
    srcs = [
        "opencl_grayscale_filter.cpp",
        ":grayscale_kernel_spv_embed",
        ":grayscale_kernel_cl_embed",
    ],
    hdrs = [
        "opencl_grayscale_filter.h",
        ":grayscale_kernel_blob_h",
    ],
    defines = ["CL_TARGET_OPENCL_VERSION=300"],
    visibility = ["//visibility:public"],
    deps = [
        ":opencl_context",
        "//src/cpp_accelerator/core:logger",
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "@opencl_sdk//:opencl",
        "@spdlog",
    ],
    linkopts = ["-l:libOpenCL.so.1"],
)

cc_library(
    name = "opencl_blur_filter",
    srcs = [
        "opencl_blur_filter.cpp",
        ":blur_kernel_spv_embed",
        ":blur_kernel_cl_embed",
    ],
    hdrs = [
        "opencl_blur_filter.h",
        ":blur_kernel_blob_h",
    ],
    defines = ["CL_TARGET_OPENCL_VERSION=300"],
    visibility = ["//visibility:public"],
    deps = [
        ":opencl_context",
        "//src/cpp_accelerator/core:logger",
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "@opencl_sdk//:opencl",
        "@spdlog",
    ],
    linkopts = ["-l:libOpenCL.so.1"],
)

cc_library(
    name = "opencl_filter_factory",
    srcs = ["opencl_filter_factory.cpp"],
    hdrs = ["opencl_filter_factory.h"],
    defines = ["CL_TARGET_OPENCL_VERSION=300"],
    visibility = ["//visibility:public"],
    deps = [
        ":opencl_blur_filter",
        ":opencl_grayscale_filter",
        "//src/cpp_accelerator/application/engine:filter_interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "//proto:common_cc_proto",
        "@opencl_sdk//:opencl",
    ],
    linkopts = ["-l:libOpenCL.so.1"],
)

cc_test(
    name = "opencl_grayscale_filter_test",
    srcs = ["opencl_grayscale_filter_test.cpp"],
    data = [
        "//data:static_images/lena.png",
    ],
    defines = ["CL_TARGET_OPENCL_VERSION=300"],
    deps = [
        ":opencl_grayscale_filter",
        ":opencl_context",
        "//src/cpp_accelerator/adapters/compute/cpu:grayscale_filter",
        "//src/cpp_accelerator/adapters/image_io:image_adapters",
        "//src/cpp_accelerator/domain/interfaces",
        "@googletest//:gtest_main",
    ],
    linkopts = ["-l:libOpenCL.so.1"],
)

cc_test(
    name = "opencl_blur_filter_test",
    srcs = ["opencl_blur_filter_test.cpp"],
    data = [
        "//data:static_images/lena.png",
    ],
    defines = ["CL_TARGET_OPENCL_VERSION=300"],
    deps = [
        ":opencl_blur_filter",
        ":opencl_context",
        "//src/cpp_accelerator/adapters/image_io:image_adapters",
        "//src/cpp_accelerator/domain/interfaces",
        "@googletest//:gtest_main",
    ],
    linkopts = ["-l:libOpenCL.so.1"],
)
