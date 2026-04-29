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
    name = "data_channel_envelope",
    srcs = ["data_channel_envelope.cpp"],
    hdrs = ["data_channel_envelope.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto:image_processor_service_cc_proto",
        "@spdlog",
    ],
)
