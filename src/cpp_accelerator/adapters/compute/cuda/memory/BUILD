load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "cuda_memory_pool",
    srcs = ["cuda_memory_pool.cpp"],
    hdrs = ["cuda_memory_pool.h"],
    copts = [
        "--expt-relaxed-constexpr",
        "-Xcompiler=-Wno-error",
    ],
    visibility = ["//visibility:public"],
    deps = [],
)
