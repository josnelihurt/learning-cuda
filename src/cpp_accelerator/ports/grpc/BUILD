load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_library(
    name = "live_video_processor_lib",
    srcs = ["live_video_processor.cpp"],
    hdrs = ["live_video_processor.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto:image_processor_service_cc_proto",
        "//src/cpp_accelerator/ports/shared_lib:processor_engine",
        "@com_github_paullouisageneau_libdatachannel//:libdatachannel",
        "@spdlog",
    ],
    linkopts = [
        "-lavcodec",
        "-lavutil",
        "-lswscale",
    ],
)

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
        "//src/cpp_accelerator/ports/shared_lib:processor_engine",
    ],
)

cc_library(
    name = "webrtc_manager_lib",
    srcs = ["webrtc_manager.cpp"],
    hdrs = ["webrtc_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":live_video_processor_lib",
        "//proto:image_processor_service_cc_proto",
        "//src/cpp_accelerator/ports/shared_lib:processor_engine",
        "@spdlog",
        "@com_github_paullouisageneau_libdatachannel//:libdatachannel",
    ],
)

cc_library(
    name = "accelerator_control_client_lib",
    srcs = ["accelerator_control_client.cpp"],
    hdrs = ["accelerator_control_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":processor_engine_contract",
        ":webrtc_manager_lib",
        "//proto:accelerator_control_cc_proto",
        "//proto:accelerator_control_grpc",
        "//proto:image_processor_service_cc_proto",
        "//proto:common_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@spdlog",
    ],
)

cc_binary(
    name = "accelerator_control_client",
    srcs = ["main_client.cpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":accelerator_control_client_lib",
        ":processor_engine_adapter_lib",
        ":webrtc_manager_lib",
        "//src/cpp_accelerator/ports/shared_lib:processor_engine",
        "//proto:image_processor_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@spdlog",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)
