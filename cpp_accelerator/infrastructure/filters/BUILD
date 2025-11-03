load("@rules_cc//cc:defs.bzl", "cc_test")

cc_test(
    name = "blur_equivalence_test",
    srcs = ["blur_equivalence_test.cpp"],
    data = [
        "//data:static_images/lena.png",
    ],
    deps = [
        "//cpp_accelerator/infrastructure/cpu:blur_filter",
        "//cpp_accelerator/infrastructure/cuda:cuda_wrappers",
        "//cpp_accelerator/domain/interfaces",
        "//cpp_accelerator/domain/interfaces/filters:filters",
        "//cpp_accelerator/infrastructure/image:image_adapters",
        "@googletest//:gtest_main",
    ],
)

