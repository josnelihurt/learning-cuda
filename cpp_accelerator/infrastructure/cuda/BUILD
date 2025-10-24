load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("@rules_cc//cc:defs.bzl", "cc_library")

# Pure CUDA kernels without spdlog dependencies
cuda_library(
    name = "cuda_kernels",
    srcs = [
        "grayscale_kernel.cu",
        "simple_kernel_pure.cu",
    ],
    hdrs = [
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
        "grayscale_processor_wrapper.cpp",
        "simple_kernel_processor_wrapper.cpp",
    ],
    hdrs = [
        "grayscale_processor.h",
        "simple_kernel_processor.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_kernels",
        "//cpp_accelerator/core:telemetry",
        "//cpp_accelerator/domain/interfaces",
        "//cpp_accelerator/domain/interfaces:processors",
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
