load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
    name = "channel_labels",
    hdrs = ["channel_labels.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "session_routing",
    srcs = ["session_routing.cpp"],
    hdrs = ["session_routing.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":channel_labels",
    ],
)

cc_library(
    name = "data_channel_framing",
    srcs = ["data_channel_framing.cpp"],
    hdrs = ["data_channel_framing.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_paullouisageneau_libdatachannel//:libdatachannel",
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
        "//src/cpp_accelerator/adapters/webrtc/protocol:filter_resolver",
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
        ":channel_labels",
        ":data_channel_framing",
        ":live_video_processor",
        ":session_routing",
        "//proto:image_processor_service_cc_proto",
        "//src/cpp_accelerator/adapters/camera:camera_hub",
        "//src/cpp_accelerator/adapters/camera:gst_camera_source",
        "//src/cpp_accelerator/adapters/compute/cuda/memory:cuda_memory_pool",
        "//src/cpp_accelerator/adapters/webrtc/protocol:data_channel_envelope",
        "//src/cpp_accelerator/adapters/webrtc/protocol:filter_resolver",
        "//src/cpp_accelerator/adapters/webrtc/sdp:sdp_utils",
        "//src/cpp_accelerator/application/engine:engine",
        "//src/cpp_accelerator/application/server_info:accelerator_label",
        "//src/cpp_accelerator/application/server_info:server_info_provider",
        "//src/cpp_accelerator/ports:i_media_session",
        "@com_github_paullouisageneau_libdatachannel//:libdatachannel",
        "@spdlog",
    ],
)
