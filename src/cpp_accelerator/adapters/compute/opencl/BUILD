load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "opencl_filter_factory",
    srcs = ["opencl_filter_factory.cpp"],
    hdrs = ["opencl_filter_factory.h"],
    defines = ["CL_TARGET_OPENCL_VERSION=300"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/cpp_accelerator/adapters/compute/opencl/filters:blur_filter",
        "//src/cpp_accelerator/adapters/compute/opencl/filters:grayscale_filter",
        "//src/cpp_accelerator/application/engine:filter_interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "//proto:common_cc_proto",
        "@opencl_sdk//:opencl",
    ],
    linkopts = ["-l:libOpenCL.so.1"],
)
