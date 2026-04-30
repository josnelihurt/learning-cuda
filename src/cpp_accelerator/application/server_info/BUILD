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
    name = "filter_parameter_mapping",
    srcs = ["filter_parameter_mapping.cpp"],
    hdrs = ["filter_parameter_mapping.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto:image_processor_service_cc_proto",
    ],
)

cc_library(
    name = "accelerator_label",
    srcs = ["accelerator_label.cpp"],
    hdrs = ["accelerator_label.h"],
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
        ":filter_parameter_mapping",
        ":i_server_info_provider",
        "//proto:image_processor_service_cc_proto",
        "//src/cpp_accelerator/application/engine:engine",
        "@spdlog",
    ],
)
