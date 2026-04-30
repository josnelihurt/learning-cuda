load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
    name = "grayscale_filter",
    srcs = [
        "grayscale_filter.cpp",
        "//src/cpp_accelerator/adapters/compute/opencl/kernels:cl_grayscale_spv_embed",
        "//src/cpp_accelerator/adapters/compute/opencl/kernels:cl_grayscale_cl_embed",
    ],
    hdrs = [
        "grayscale_filter.h",
        "//src/cpp_accelerator/adapters/compute/opencl/kernels:cl_grayscale_blob_h",
    ],
    defines = ["CL_TARGET_OPENCL_VERSION=300"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/cpp_accelerator/adapters/compute/opencl/context",
        "//src/cpp_accelerator/core:logger",
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "@opencl_sdk//:opencl",
        "@spdlog",
    ],
    linkopts = ["-l:libOpenCL.so.1"],
)

cc_library(
    name = "blur_filter",
    srcs = [
        "blur_filter.cpp",
        "//src/cpp_accelerator/adapters/compute/opencl/kernels:cl_blur_spv_embed",
        "//src/cpp_accelerator/adapters/compute/opencl/kernels:cl_blur_cl_embed",
    ],
    hdrs = [
        "blur_filter.h",
        "//src/cpp_accelerator/adapters/compute/opencl/kernels:cl_blur_blob_h",
    ],
    defines = ["CL_TARGET_OPENCL_VERSION=300"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/cpp_accelerator/adapters/compute/opencl/context",
        "//src/cpp_accelerator/core:logger",
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "@opencl_sdk//:opencl",
        "@spdlog",
    ],
    linkopts = ["-l:libOpenCL.so.1"],
)

cc_test(
    name = "grayscale_filter_test",
    srcs = ["grayscale_filter_test.cpp"],
    data = [
        "//data:static_images/lena.png",
    ],
    defines = ["CL_TARGET_OPENCL_VERSION=300"],
    deps = [
        ":grayscale_filter",
        "//src/cpp_accelerator/adapters/compute/opencl/context",
        "//src/cpp_accelerator/adapters/compute/cpu:grayscale_filter",
        "//src/cpp_accelerator/adapters/image_io:image_adapters",
        "//src/cpp_accelerator/domain/interfaces",
        "@googletest//:gtest_main",
    ],
    linkopts = ["-l:libOpenCL.so.1"],
)

cc_test(
    name = "blur_filter_test",
    srcs = ["blur_filter_test.cpp"],
    data = [
        "//data:static_images/lena.png",
    ],
    defines = ["CL_TARGET_OPENCL_VERSION=300"],
    deps = [
        ":blur_filter",
        "//src/cpp_accelerator/adapters/compute/opencl/context",
        "//src/cpp_accelerator/adapters/image_io:image_adapters",
        "//src/cpp_accelerator/domain/interfaces",
        "@googletest//:gtest_main",
    ],
    linkopts = ["-l:libOpenCL.so.1"],
)
