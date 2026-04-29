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
    name = "engine",
    srcs = ["processor_engine.cpp"],
    hdrs = ["processor_engine.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":filter_factory_registry",
        ":filter_interfaces",
        "//src/cpp_accelerator/application/pipeline:filter_pipeline",
        "//src/cpp_accelerator/core:logger",
        "//src/cpp_accelerator/core:telemetry",
        "//src/cpp_accelerator/core:version",
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces:i_yolo_detector",
        "//src/cpp_accelerator/adapters/compute/cpu:cpu_filter_factory",
        "//src/cpp_accelerator/adapters/compute/cuda:cuda_filter_factory",
        "//src/cpp_accelerator/adapters/compute/opencl:opencl_filter_factory",
        "//src/cpp_accelerator/adapters/compute/cuda/tensorrt:model_manager",
        "//proto:common_cc_proto",
        "//proto:image_processor_service_cc_proto",
        "@spdlog",
    ],
    linkopts = ["-l:libOpenCL.so.1"],
)
