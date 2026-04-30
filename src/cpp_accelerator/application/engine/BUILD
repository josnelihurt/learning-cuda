load("@bazel_skylib//lib:selects.bzl", "selects")

cc_library(
    name = "filter_interfaces",
    hdrs = [
        "filter_creation_dispatch.hpp",
        "filter_descriptor.h",
        "i_filter_factory.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "//proto:common_cc_proto",
    ],
)

cc_library(
    name = "filter_factory_registry",
    srcs = ["filter_factory_registry.cpp"],
    hdrs = ["filter_factory_registry.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":filter_interfaces",
        "//proto:common_cc_proto",
    ],
)

cc_library(
    name = "cuda_platform_impl",
    srcs = ["platform/cuda/cuda_platform.cpp"],
    hdrs = ["platform/cuda/cuda_platform.h"],
    deps = [
        ":filter_factory_registry",
        ":filter_interfaces",
        "//src/cpp_accelerator/adapters/compute/cuda:cuda_filter_factory",
        "//src/cpp_accelerator/adapters/compute/cuda/tensorrt:model_manager",
        "//src/cpp_accelerator/domain/interfaces:i_yolo_detector",
        "//proto:common_cc_proto",
        "//proto:image_processor_service_cc_proto",
        "@spdlog",
    ],
)

cc_library(
    name = "opencl_platform_impl",
    srcs = ["platform/opencl/opencl_platform.cpp"],
    hdrs = ["platform/opencl/opencl_platform.h"],
    linkopts = ["-l:libOpenCL.so.1"],
    deps = [
        ":filter_factory_registry",
        ":filter_interfaces",
        "//src/cpp_accelerator/adapters/compute/opencl:opencl_filter_factory",
        "//proto:common_cc_proto",
    ],
)

cc_library(
    name = "vulkan_platform_impl",
    srcs = ["platform/vulkan/vulkan_platform.cpp"],
    hdrs = ["platform/vulkan/vulkan_platform.h"],
    linkopts = ["-lvulkan"],
    deps = [
        ":filter_factory_registry",
        ":filter_interfaces",
        "//src/cpp_accelerator/adapters/compute/vulkan:vulkan_filter_factory",
        "//proto:common_cc_proto",
    ],
)

selects.config_setting_group(
    name = "cuda_and_opencl",
    match_all = ["//bazel/flags:cuda_enabled", "//bazel/flags:opencl_enabled"],
)

selects.config_setting_group(
    name = "cuda_and_vulkan",
    match_all = ["//bazel/flags:cuda_enabled", "//bazel/flags:vulkan_enabled"],
)

selects.config_setting_group(
    name = "opencl_and_vulkan",
    match_all = ["//bazel/flags:opencl_enabled", "//bazel/flags:vulkan_enabled"],
)

selects.config_setting_group(
    name = "cuda_opencl_and_vulkan",
    match_all = [
        "//bazel/flags:cuda_enabled",
        "//bazel/flags:opencl_enabled",
        "//bazel/flags:vulkan_enabled",
    ],
)

cc_library(
    name = "platform_support",
    hdrs = ["platform/platform_support.h"],
    srcs = select({
        ":cuda_opencl_and_vulkan":      ["platform/platform_support_all.cpp"],
        ":cuda_and_opencl":             ["platform/platform_support_full.cpp"],
        ":cuda_and_vulkan":             ["platform/platform_support_cuda_vulkan.cpp"],
        ":opencl_and_vulkan":           ["platform/platform_support_opencl_vulkan.cpp"],
        "//bazel/flags:cuda_enabled":   ["platform/platform_support_cuda.cpp"],
        "//bazel/flags:opencl_enabled": ["platform/platform_support_opencl.cpp"],
        "//bazel/flags:vulkan_enabled": ["platform/platform_support_vulkan.cpp"],
        "//conditions:default":         ["platform/platform_support_cpu.cpp"],
    }),
    deps = [
        ":filter_factory_registry",
        ":filter_interfaces",
        "//src/cpp_accelerator/adapters/compute/cpu:cpu_filter_factory",
        "//src/cpp_accelerator/domain/interfaces",
        "//proto:common_cc_proto",
        "//proto:image_processor_service_cc_proto",
    ] + select({
        ":cuda_opencl_and_vulkan":      [":cuda_platform_impl", ":opencl_platform_impl", ":vulkan_platform_impl"],
        ":cuda_and_opencl":             [":cuda_platform_impl", ":opencl_platform_impl"],
        ":cuda_and_vulkan":             [":cuda_platform_impl", ":vulkan_platform_impl"],
        ":opencl_and_vulkan":           [":opencl_platform_impl", ":vulkan_platform_impl"],
        "//bazel/flags:cuda_enabled":   [":cuda_platform_impl"],
        "//bazel/flags:opencl_enabled": [":opencl_platform_impl"],
        "//bazel/flags:vulkan_enabled": [":vulkan_platform_impl"],
        "//conditions:default":         [],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "engine",
    srcs = ["processor_engine.cpp"],
    hdrs = ["processor_engine.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":filter_factory_registry",
        ":filter_interfaces",
        ":platform_support",
        "//src/cpp_accelerator/application/pipeline:filter_pipeline",
        "//src/cpp_accelerator/core:logger",
        "//src/cpp_accelerator/core:telemetry",
        "//src/cpp_accelerator/core:version",
        "//src/cpp_accelerator/domain/interfaces",
        "//proto:common_cc_proto",
        "//proto:image_processor_service_cc_proto",
        "@spdlog",
    ],
)
