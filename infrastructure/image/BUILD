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
        "//domain/interfaces:interfaces",
        "//third_party/stb:stb_image",
        "//third_party/stb:stb_image_write",
        "@spdlog//:spdlog",
    ],
)

