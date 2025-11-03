cc_library(
    name = "grayscale_processor",
    srcs = ["grayscale_processor.cpp"],
    hdrs = ["grayscale_processor.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//cpp_accelerator/core:telemetry",
        "//cpp_accelerator/domain/interfaces",
        "//cpp_accelerator/domain/interfaces:processors",
        "@spdlog",
    ],
)

cc_library(
    name = "blur_filter",
    srcs = ["blur_filter.cpp"],
    hdrs = ["blur_filter.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//cpp_accelerator/domain/interfaces",
        "//cpp_accelerator/domain/interfaces/filters:filters",
    ],
)

cc_library(
    name = "grayscale_filter",
    srcs = ["grayscale_filter.cpp"],
    hdrs = ["grayscale_filter.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//cpp_accelerator/domain/interfaces",
        "//cpp_accelerator/domain/interfaces/filters:filters",
        ":grayscale_processor",
    ],
)

cc_test(
    name = "blur_filter_test",
    srcs = ["blur_filter_test.cpp"],
    data = [
        "//data:static_images/lena.png",
    ],
    deps = [
        ":blur_filter",
        "//cpp_accelerator/domain/interfaces",
        "//cpp_accelerator/domain/interfaces/filters:filters",
        "//cpp_accelerator/infrastructure/image:image_adapters",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "grayscale_processor_test",
    srcs = ["grayscale_processor_test.cpp"],
    data = [
        "//data:static_images/lena.png",
    ],
    deps = [
        ":grayscale_processor",
        "//cpp_accelerator/domain/interfaces",
        "//cpp_accelerator/domain/interfaces:processors",
        "//cpp_accelerator/infrastructure/image:image_adapters",
        "@googletest//:gtest_main",
    ],
)
