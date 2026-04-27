load("@rules_cc//cc:defs.bzl", "cc_library")

config_setting(
    name = "linux_aarch64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
)

cc_library(
    name = "tensorrt",
    linkopts = [
        "-lnvinfer",
        "-lnvonnxparser",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "yolo_detector",
    srcs = ["yolo_detector.cpp"],
    hdrs = ["yolo_detector.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tensorrt",
        "//src/cpp_accelerator/adapters/compute/cuda/kernels:kernels",
        "//src/cpp_accelerator/domain/interfaces:i_yolo_detector",
        "//src/cpp_accelerator/core:telemetry",
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "@spdlog",
    ],
)

cc_library(
    name = "yolo_factory",
    hdrs = ["yolo_factory.h"],
    srcs = ["yolo_factory_trt.cpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/cpp_accelerator/domain/interfaces:i_yolo_detector",
        ":yolo_detector",
    ],
)

cc_library(
    name = "model_registry",
    srcs = ["model_registry.cpp"],
    hdrs = ["model_registry.h"],
    visibility = ["//visibility:public"],
    deps = [],
)

cc_library(
    name = "model_manager",
    srcs = ["model_manager.cpp"],
    hdrs = ["model_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/cpp_accelerator/domain/interfaces:i_yolo_detector",
        ":model_registry",
        ":yolo_factory",
        "@spdlog",
    ],
)
