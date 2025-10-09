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
        "//cpp_accelerator/domain/interfaces:interfaces",
        "//third_party/stb:stb_image",
        "//third_party/stb:stb_image_write",
        "@spdlog//:spdlog",
    ],
)

cc_test(
    name = "image_adapters_test",
    srcs = [
        "image_loader_test.cpp",
        "image_writer_test.cpp",
    ],
    data = [
        "//data:lena.png",
    ],
    deps = [
        ":image_adapters",
        "//cpp_accelerator/domain/interfaces:interfaces",
        "@googletest//:gtest_main",
    ],
)

