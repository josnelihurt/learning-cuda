load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
    name = "data_channel_framing",
    srcs = ["data_channel_framing.cpp"],
    hdrs = ["data_channel_framing.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@spdlog",
    ],
)

cc_test(
    name = "data_channel_framing_test",
    srcs = ["data_channel_framing_test.cpp"],
    deps = [
        ":data_channel_framing",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "live_video_processor",
    srcs = ["live_video_processor.cpp"],
    hdrs = ["live_video_processor.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto:image_processor_service_cc_proto",
        "//src/cpp_accelerator/application/engine:engine",
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
    name = "webrtc_manager",
    srcs = ["webrtc_manager.cpp"],
    hdrs = ["webrtc_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":data_channel_framing",
        ":live_video_processor",
        "//proto:image_processor_service_cc_proto",
        "//src/cpp_accelerator/application/engine:engine",
        "//src/cpp_accelerator/adapters/compute/cuda/memory:cuda_memory_pool",
        "@spdlog",
        "@com_github_paullouisageneau_libdatachannel//:libdatachannel",
    ],
)
