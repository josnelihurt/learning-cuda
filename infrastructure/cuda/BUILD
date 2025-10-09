load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "cuda_processors",
    srcs = [
        "simple_kernel.cu",
        "simple_kernel_processor.cu",
        "grayscale_processor.cu",
    ],
    hdrs = [
        "simple_kernel.h",
        "simple_kernel_processor.h",
        "grayscale_processor.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//domain/interfaces:interfaces",
        "//domain/interfaces:processors",
        "@spdlog//:spdlog",
    ],
)

