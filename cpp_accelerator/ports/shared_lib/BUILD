cc_library(
    name = "image_buffer_adapter",
    srcs = ["image_buffer_adapter.cpp"],
    hdrs = ["image_buffer_adapter.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//cpp_accelerator/domain/interfaces",
    ],
)

cc_binary(
    name = "libcuda_processor.so",
    srcs = [
        "cuda_processor_impl.cpp",
        "processor_api.h",
    ],
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [
        ":image_buffer_adapter",
        "//cpp_accelerator/core:logger",
        "//cpp_accelerator/core:telemetry",
        "//cpp_accelerator/infrastructure/cpu:grayscale_processor",
        "//cpp_accelerator/infrastructure/cuda:cuda_processors",
        "//proto:common_cc_proto",
        "//proto:image_processor_service_cc_proto",
        "@spdlog",
    ],
)

