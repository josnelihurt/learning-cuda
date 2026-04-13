cc_library(
    name = "image_buffer_adapter",
    srcs = ["image_buffer_adapter.cpp"],
    hdrs = ["image_buffer_adapter.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//cpp_accelerator/domain/interfaces",
    ],
)

cc_library(
    name = "processor_api",
    hdrs = ["processor_api.h"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "library_version_private",
    srcs = ["//:cpp_accelerator/VERSION"],
    outs = ["library_version_private.h"],
    cmd = "VERSION=$$(cat $(location //:cpp_accelerator/VERSION) | tr -d '[:space:]'); echo '#pragma once' > $@; echo '#define LIBRARY_VERSION \"'$$VERSION'\"' >> $@",
)

cc_library(
    name = "library_version",
    hdrs = [
        "library_version.h",
        ":library_version_private",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "processor_engine",
    srcs = ["processor_engine.cpp"],
    hdrs = ["processor_engine.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":library_version",
        ":processor_api",
        "//cpp_accelerator/application/pipeline:filter_pipeline",
        "//cpp_accelerator/core:logger",
        "//cpp_accelerator/core:telemetry",
        "//cpp_accelerator/domain/interfaces",
        "//cpp_accelerator/infrastructure/cpu:blur_filter",
        "//cpp_accelerator/infrastructure/cpu:grayscale_filter",
        "//cpp_accelerator/infrastructure/cuda:blur_processor",
        "//cpp_accelerator/infrastructure/cuda:grayscale_filter",
        "//proto:common_cc_proto",
        "//proto:image_processor_service_cc_proto",
        "@spdlog",
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
        ":library_version",
        ":processor_engine",
        "//cpp_accelerator/core:logger",
        "//cpp_accelerator/core:telemetry",
        "//proto:common_cc_proto",
        "//proto:image_processor_service_cc_proto",
        "@spdlog",
    ],
)

cc_test(
    name = "blur_e2e_test",
    srcs = [
        "blur_e2e_test.cpp",
        "cuda_processor_impl.cpp",
    ],
    data = [
        "//data:static_images/lena.png",
    ],
    deps = [
        ":image_buffer_adapter",
        ":processor_api",
        ":processor_engine",
        "//cpp_accelerator/application/pipeline:filter_pipeline",
        "//cpp_accelerator/core:logger",
        "//cpp_accelerator/core:telemetry",
        "@googletest//:gtest_main",
        "@spdlog",
        "//cpp_accelerator/domain/interfaces",
        "//cpp_accelerator/infrastructure/cpu:blur_filter",
        "//cpp_accelerator/infrastructure/cpu:grayscale_filter",
        "//cpp_accelerator/infrastructure/cuda:cuda_kernels",
        "//cpp_accelerator/infrastructure/cuda:cuda_wrappers",
        "//cpp_accelerator/infrastructure/cuda:grayscale_filter",
        "//cpp_accelerator/infrastructure/image:image_adapters",
        "//proto:common_cc_proto",
        "//proto:image_processor_service_cc_proto",
    ],
)

