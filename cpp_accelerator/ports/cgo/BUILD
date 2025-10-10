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
    name = "cgo_api",
    srcs = ["cgo_api.cpp"],
    hdrs = ["cgo_api.h"],
    alwayslink = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":image_buffer_adapter",
        "//cpp_accelerator/infrastructure/cuda:cuda_processors",
        "//proto:image_processing_cc_proto",
        "@spdlog",
    ],
)

