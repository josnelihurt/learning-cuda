load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "context",
    srcs = ["context.cpp"],
    hdrs = ["context.h"],
    defines = ["CL_TARGET_OPENCL_VERSION=300"],
    visibility = ["//src/cpp_accelerator/adapters/compute/opencl:__subpackages__"],
    deps = [
        "//src/cpp_accelerator/core:logger",
        "@opencl_sdk//:opencl",
        "@spdlog",
    ],
    linkopts = ["-l:libOpenCL.so.1"],
)
