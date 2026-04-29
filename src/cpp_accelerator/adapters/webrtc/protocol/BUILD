load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "filter_resolver",
    srcs = ["filter_resolver.cpp"],
    hdrs = ["filter_resolver.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto:image_processor_service_cc_proto",
        "@spdlog",
    ],
)

cc_library(
    name = "message_codec",
    srcs = ["message_codec.cpp"],
    hdrs = ["message_codec.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto:image_processor_service_cc_proto",
        "//src/cpp_accelerator/adapters/webrtc:data_channel_framing",
        "@com_github_paullouisageneau_libdatachannel//:libdatachannel",
        "@spdlog",
    ],
)

cc_library(
    name = "session_routing",
    srcs = ["session_routing.cpp"],
    hdrs = ["session_routing.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto:image_processor_service_cc_proto",
    ],
)
