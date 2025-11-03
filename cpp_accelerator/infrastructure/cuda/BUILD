load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("@rules_cc//cc:defs.bzl", "cc_library")

# Pure CUDA kernels without spdlog dependencies
cuda_library(
    name = "cuda_kernels",
    srcs = [
        "blur_kernel.cu",
        "grayscale_kernel.cu",
        "simple_kernel_pure.cu",
    ],
    hdrs = [
        "blur_kernel.h",
        "grayscale_kernel.h",
        "simple_kernel_pure.h",
    ],
    copts = [
        "--expt-relaxed-constexpr",
        "-Xcompiler=-Wno-error",
    ],
    visibility = ["//visibility:public"],
    deps = [],
)

# C++ wrappers that handle spdlog and telemetry
cc_library(
    name = "cuda_wrappers",
    srcs = [
        "blur_processor_wrapper.cpp",
        "grayscale_processor_wrapper.cpp",
        "simple_kernel_processor_wrapper.cpp",
    ],
    hdrs = [
        "blur_processor.h",
        "grayscale_processor.h",
        "simple_kernel_processor.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_kernels",
        "//cpp_accelerator/core:telemetry",
        "//cpp_accelerator/domain/interfaces",
        "//cpp_accelerator/domain/interfaces:processors",
        "//cpp_accelerator/domain/interfaces/filters:filters",
        "@spdlog",
    ],
)

# Combined library for backward compatibility
cc_library(
    name = "cuda_processors",
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_kernels",
        ":cuda_wrappers",
    ],
)

cc_test(
    name = "blur_processor_test",
    srcs = ["blur_processor_test.cpp"],
    data = [
        "//data:static_images/lena.png",
    ],
    deps = [
        ":cuda_wrappers",
        "//cpp_accelerator/domain/interfaces",
        "//cpp_accelerator/domain/interfaces/filters:filters",
        "//cpp_accelerator/infrastructure/image:image_adapters",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "grayscale_processor_test",
    srcs = ["grayscale_processor_test.cpp"],
    data = [
        "//data:static_images/lena.png",
    ],
    deps = [
        ":cuda_wrappers",
        "//cpp_accelerator/domain/interfaces",
        "//cpp_accelerator/domain/interfaces:processors",
        "//cpp_accelerator/infrastructure/image:image_adapters",
        "@googletest//:gtest_main",
    ],
)
