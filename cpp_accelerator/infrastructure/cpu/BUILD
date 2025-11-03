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
    name = "grayscale_filter_test",
    srcs = ["grayscale_filter_test.cpp"],
    data = [
        "//data:static_images/lena.png",
    ],
    deps = [
        ":grayscale_filter",
        "//cpp_accelerator/domain/interfaces",
        "//cpp_accelerator/domain/interfaces/filters:filters",
        "//cpp_accelerator/infrastructure/image:image_adapters",
        "@googletest//:gtest_main",
    ],
)
