cc_library(
    name = "blur_filter",
    srcs = ["blur_filter.cpp"],
    hdrs = ["blur_filter.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
    ],
)

cc_library(
    name = "grayscale_filter",
    srcs = ["grayscale_filter.cpp"],
    hdrs = ["grayscale_filter.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
    ],
)

cc_library(
    name = "cpu_filter_factory",
    srcs = ["cpu_filter_factory.cpp"],
    hdrs = ["cpu_filter_factory.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":blur_filter",
        ":grayscale_filter",
        "//src/cpp_accelerator/application/engine:filter_interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "//proto:common_cc_proto",
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
        "//src/cpp_accelerator/domain/interfaces",
        "//src/cpp_accelerator/domain/interfaces/filters:filters",
        "//src/cpp_accelerator/adapters/image_io:image_adapters",
        "@googletest//:gtest_main",
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
        "//src/cpp_accelerator/adapters/image_io:image_adapters",
        "@googletest//:gtest_main",
    ],
)
