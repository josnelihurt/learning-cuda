load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("@rules_cc//cc:defs.bzl", "cc_library")

config_setting(
    name = "linux_aarch64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
)

# Pure CUDA kernels without spdlog dependencies
cuda_library(
    name = "cuda_kernels",
    srcs = [
        "blur_kernel.cu",
        "grayscale_kernel.cu",
        "letterbox_kernel.cu",
    ],
    hdrs = [
        "blur_kernel.h",
        "grayscale_kernel.h",
        "letterbox_kernel.h",
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
    name = "blur_processor",
    srcs = [
        "blur_processor_wrapper.cpp",
    ],
    hdrs = [
        "blur_processor.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_kernels",
        "//src/cpp_accelerator/core:telemetry",
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces:processors",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "@spdlog",
    ],
)

cc_library(
    name = "cuda_wrappers",
    hdrs = ["blur_processor.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":blur_processor",
    ],
)

# Combined library for backward compatibility
cc_library(
    name = "cuda_processors",
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_kernels",
        ":blur_processor",
    ],
)

cc_test(
    name = "blur_processor_test",
    srcs = ["blur_processor_test.cpp"],
    data = [
        "//data:static_images/lena.png",
    ],
    deps = [
        ":blur_processor",
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "//src/cpp_accelerator/infrastructure/image:image_adapters",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "grayscale_filter",
    srcs = ["grayscale_filter.cpp"],
    hdrs = ["grayscale_filter.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_kernels",
        "//src/cpp_accelerator/core:telemetry",
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "@spdlog",
    ],
)

cc_test(
    name = "grayscale_filter_test",
    srcs = ["grayscale_filter_test.cpp"],
    data = [
        "//data:static_images/lena.png",
    ],
    deps = [
        ":grayscale_filter",
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "//src/cpp_accelerator/infrastructure/cpu:grayscale_filter",
        "//src/cpp_accelerator/infrastructure/image:image_adapters",
        "@googletest//:gtest_main",
    ],
)

# TensorRT libraries for YOLO object detection
# Assumes TensorRT is installed system-wide at /usr/local/lib
cc_library(
    name = "tensorrt",
    linkopts = [
        "-lnvinfer",
        "-lnvonnxparser",
    ],
    visibility = ["//visibility:public"],
)

# YOLO Object Detector (TensorRT implementation)
cc_library(
    name = "yolo_detector",
    srcs = ["yolo_detector.cpp"],
    hdrs = ["yolo_detector.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_kernels",
        ":detection_hdr",
        ":i_yolo_detector",
        ":tensorrt",
        "//src/cpp_accelerator/core:telemetry",
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "@spdlog",
    ],
)

# Shared Detection struct header
cc_library(
    name = "detection_hdr",
    hdrs = ["detection.h"],
    visibility = ["//visibility:public"],
)

# IYoloDetector abstract interface
cc_library(
    name = "i_yolo_detector",
    hdrs = ["i_yolo_detector.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":detection_hdr",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
    ],
)

# YOLO detector factory
cc_library(
    name = "yolo_factory",
    hdrs = ["yolo_factory.h"],
    srcs = ["yolo_factory_trt.cpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":i_yolo_detector",
        ":yolo_detector",
    ],
)

# Model Registry for AI/ML models
cc_library(
    name = "model_registry",
    srcs = ["model_registry.cpp"],
    hdrs = ["model_registry.h"],
    visibility = ["//visibility:public"],
    deps = [],
)

# Model Manager for preloading and caching detectors
cc_library(
    name = "model_manager",
    srcs = ["model_manager.cpp"],
    hdrs = ["model_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":i_yolo_detector",
        ":model_registry",
        ":yolo_factory",
        "@spdlog",
    ],
)
