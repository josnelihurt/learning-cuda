load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

cc_library(
    name = "processor_engine_contract",
    hdrs = ["processor_engine_provider.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto:image_processor_service_cc_proto",
    ],
)

cc_library(
    name = "processor_engine_adapter_lib",
    srcs = ["processor_engine_adapter.cpp"],
    hdrs = ["processor_engine_adapter.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":processor_engine_contract",
        "//cpp_accelerator/ports/shared_lib:processor_engine",
    ],
)

cc_library(
    name = "image_processor_service_impl_lib",
    srcs = ["image_processor_service_impl.cpp"],
    hdrs = ["image_processor_service_impl.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":processor_engine_contract",
        "//proto:image_processor_service_cc_proto",
        "//proto:image_processor_service_grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@spdlog",
    ],
)

cc_binary(
    name = "image_processor_grpc_server",
    srcs = ["server_main.cpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":image_processor_service_impl_lib",
        ":processor_engine_adapter_lib",
        "//cpp_accelerator/ports/shared_lib:processor_engine",
        "//proto:image_processor_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@spdlog",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings:strings",
    ],
)

cc_test(
    name = "image_processor_service_impl_test",
    srcs = ["image_processor_service_impl_test.cpp"],
    linkstatic = False,
    deps = [
        ":image_processor_service_impl_lib",
        ":processor_engine_contract",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/strings:strings",
        "@googletest//:gtest_main",
    ],
)

