cc_library(
    name = "image_adapters",
    srcs = [
        "image_loader.cpp",
        "image_writer.cpp",
    ],
    hdrs = [
        "image_loader.h",
        "image_writer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//cpp_accelerator/domain/interfaces",
        "@spdlog",
        "//third_party/stb:stb_image",
        "//third_party/stb:stb_image_write",
    ],
)

cc_test(
    name = "image_adapters_test",
    srcs = [
        "image_loader_test.cpp",
        "image_writer_test.cpp",
    ],
    data = [
        "//data:static_images/lena.png",
    ],
    deps = [
        ":image_adapters",
        "//cpp_accelerator/domain/interfaces",
        "@googletest//:gtest_main",
    ],
)
