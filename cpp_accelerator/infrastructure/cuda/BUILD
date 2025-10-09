load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "cuda_processors",
    srcs = [
        "grayscale_processor.cu",
        "simple_kernel.cu",
        "simple_kernel_processor.cu",
    ],
    hdrs = [
        "grayscale_processor.h",
        "simple_kernel.h",
        "simple_kernel_processor.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//cpp_accelerator/domain/interfaces",
        "//cpp_accelerator/domain/interfaces:processors",
        "@spdlog",
    ],
)
