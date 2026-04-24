cc_library(
    name = "image_buffer_adapter",
    srcs = ["image_buffer_adapter.cpp"],
    hdrs = ["image_buffer_adapter.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/cpp_accelerator/domain/interfaces",
    ],
)

cc_library(
    name = "processor_api",
    hdrs = ["processor_api.h"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "library_version_private",
    srcs = ["//:src/cpp_accelerator/VERSION"],
    outs = ["library_version_private.h"],
    cmd = "VERSION=$$(cat $(location //:src/cpp_accelerator/VERSION) | tr -d '[:space:]'); echo '#pragma once' > $@; echo '#define LIBRARY_VERSION \"'$$VERSION'\"' >> $@",
)

genrule(
    name = "git_hash_private",
    outs = ["git_hash_private.h"],
    cmd = "HASH=$$${LIBRARY_GIT_HASH:-$$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')}; echo '#pragma once' > $@; echo '#define LIBRARY_GIT_HASH \"'$$HASH'\"' >> $@",
)

cc_library(
    name = "git_hash",
    hdrs = [":git_hash_private"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "library_version",
    hdrs = [
        "library_version.h",
        ":git_hash_private",
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
        "//src/cpp_accelerator/application/pipeline:filter_pipeline",
        "//src/cpp_accelerator/core:logger",
        "//src/cpp_accelerator/core:telemetry",
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/infrastructure/cpu:blur_filter",
        "//src/cpp_accelerator/infrastructure/cpu:grayscale_filter",
        "//src/cpp_accelerator/infrastructure/cuda:blur_processor",
        "//src/cpp_accelerator/infrastructure/cuda:grayscale_filter",
        "//src/cpp_accelerator/infrastructure/cuda:i_yolo_detector",
        "//src/cpp_accelerator/infrastructure/cuda:model_manager",
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
        "//src/cpp_accelerator/core:logger",
        "//src/cpp_accelerator/core:telemetry",
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
        "//src/cpp_accelerator/application/pipeline:filter_pipeline",
        "//src/cpp_accelerator/core:logger",
        "//src/cpp_accelerator/core:telemetry",
        "@googletest//:gtest_main",
        "@spdlog",
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/infrastructure/cpu:blur_filter",
        "//src/cpp_accelerator/infrastructure/cpu:grayscale_filter",
        "//src/cpp_accelerator/infrastructure/cuda:cuda_kernels",
        "//src/cpp_accelerator/infrastructure/cuda:cuda_wrappers",
        "//src/cpp_accelerator/infrastructure/cuda:grayscale_filter",
        "//src/cpp_accelerator/infrastructure/image:image_adapters",
        "//proto:common_cc_proto",
        "//proto:image_processor_service_cc_proto",
    ],
)

