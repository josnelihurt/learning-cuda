cc_library(
    name = "cpu_platform_impl",
    srcs = ["platform/cpu/cpu_platform.cpp"],
    hdrs = ["platform/cpu/cpu_platform.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//src/cpp_accelerator/adapters/compute/cpu:cpu_filter_factory",
        "//src/cpp_accelerator/application/engine:filter_factory_registry",
        "//src/cpp_accelerator/application/engine:filter_interfaces",
        "//proto:common_cc_proto",
    ],
)

cc_library(
    name = "cuda_platform_impl",
    srcs = ["platform/cuda/cuda_platform.cpp"],
    hdrs = ["platform/cuda/cuda_platform.h"],
    deps = [
        "//src/cpp_accelerator/adapters/compute/cuda:cuda_filter_factory",
        "//src/cpp_accelerator/adapters/compute/cuda/tensorrt:model_manager",
        "//src/cpp_accelerator/application/engine:filter_factory_registry",
        "//src/cpp_accelerator/application/engine:filter_interfaces",
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
        "//src/cpp_accelerator/adapters/compute/opencl:opencl_filter_factory",
        "//src/cpp_accelerator/application/engine:filter_factory_registry",
        "//src/cpp_accelerator/application/engine:filter_interfaces",
        "//proto:common_cc_proto",
    ],
)

cc_library(
    name = "vulkan_platform_impl",
    srcs = ["platform/vulkan/vulkan_platform.cpp"],
    hdrs = ["platform/vulkan/vulkan_platform.h"],
    linkopts = ["-lvulkan"],
    deps = [
        "//src/cpp_accelerator/adapters/compute/vulkan:vulkan_filter_factory",
        "//src/cpp_accelerator/application/engine:filter_factory_registry",
        "//src/cpp_accelerator/application/engine:filter_interfaces",
        "//proto:common_cc_proto",
    ],
)

config_setting(
    name = "cuda_and_opencl",
    flag_values = {
        "//bazel/flags:cuda": "true",
        "//bazel/flags:opencl": "true",
    },
)

config_setting(
    name = "cuda_and_vulkan",
    flag_values = {
        "//bazel/flags:cuda": "true",
        "//bazel/flags:vulkan": "true",
    },
)

config_setting(
    name = "opencl_and_vulkan",
    flag_values = {
        "//bazel/flags:opencl": "true",
        "//bazel/flags:vulkan": "true",
    },
)

config_setting(
    name = "cuda_opencl_and_vulkan",
    flag_values = {
        "//bazel/flags:cuda": "true",
        "//bazel/flags:opencl": "true",
        "//bazel/flags:vulkan": "true",
    },
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
        ":cpu_platform_impl",
        "//src/cpp_accelerator/application/engine:filter_factory_registry",
        "//src/cpp_accelerator/application/engine:filter_interfaces",
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
