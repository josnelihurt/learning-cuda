load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "i_server_info_provider",
    hdrs = ["i_server_info_provider.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto:image_processor_service_cc_proto",
    ],
)

cc_library(
    name = "server_info_provider",
    srcs = ["server_info_provider.cpp"],
    hdrs = ["server_info_provider.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":i_server_info_provider",
        "//proto:image_processor_service_cc_proto",
        "//src/cpp_accelerator/adapters/webrtc/protocol:filter_resolver",
        "//src/cpp_accelerator/application/engine:engine",
        "@spdlog",
    ],
)
