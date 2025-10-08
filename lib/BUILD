load("@rules_cuda//cuda:defs.bzl", "cuda_library")

# Main library with all components
cuda_library(
    name = "lib",
    srcs = [
        "cuda/simple_kernel.cu",
        "cuda/grayscale_processor.cu",
        "image/image_loader.cpp",
        "image/image_writer.cpp",
    ],
    hdrs = [
        "cuda/simple_kernel.h",
        "cuda/grayscale_processor.h",
        "image/image_loader.h",
        "image/image_writer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//interfaces:interfaces",
        "//third_party/stb:stb_image",
        "//third_party/stb:stb_image_write",
        "@spdlog//:spdlog",
    ],
)
