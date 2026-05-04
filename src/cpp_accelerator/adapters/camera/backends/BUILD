load("@rules_cc//cc:defs.bzl", "cc_library")

# Camera backend interface
cc_library(
    name = "camera_backend",
    hdrs = ["camera_backend.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto:accelerator_control_cc_proto",
        "@com_github_paullouisageneau_libdatachannel//:libdatachannel",
    ],
)

# V4L2 backend (Linux V4L2 + GStreamer)
cc_library(
    name = "v4l2_backend",
    srcs = select({
        "//bazel/flags:v4l2_camera_enabled": ["v4l2_backend.cpp"],
        "//conditions:default": [],
    }),
    hdrs = ["v4l2_backend.h"],
    visibility = ["//visibility:public"],
    copts = select({
        "//bazel/flags:v4l2_camera_enabled": [
            "-I/usr/include/gstreamer-1.0",
            "-I/usr/include/glib-2.0",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":camera_backend",
        "@spdlog",
    ] + select({
        "//bazel/flags:v4l2_camera_enabled": [
            "//third_party/glib-config:glibconfig",
            "//third_party/gstreamer:gstreamer",
        ],
        "//conditions:default": [],
    }),
)

# NVIDIA Argus backend (GStreamer nvarguscamerasrc)
cc_library(
    name = "nvidia_argus_backend",
    srcs = select({
        "//bazel/flags:nvidia_argus_camera_enabled": ["nvidia_argus_backend.cpp"],
        "//conditions:default": [],
    }),
    hdrs = ["nvidia_argus_backend.h"],
    visibility = ["//visibility:public"],
    copts = select({
        "//bazel/flags:nvidia_argus_camera_enabled": [
            "-I/usr/include/gstreamer-1.0",
            "-I/usr/include/glib-2.0",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":camera_backend",
        "//src/cpp_accelerator/adapters/camera:gpu_frame_processor",
        "@com_github_paullouisageneau_libdatachannel//:libdatachannel",
        "@spdlog",
    ] + select({
        "//bazel/flags:nvidia_argus_camera_enabled": [
            "//third_party/glib-config:glibconfig",
            "//third_party/gstreamer:gstreamer",
        ],
        "//conditions:default": [],
    }),
)

# Stub backend (fallback when no backends are enabled)
cc_library(
    name = "stub_backend",
    srcs = ["stub_backend.cpp"],
    hdrs = ["stub_backend.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":camera_backend",
        "@spdlog",
    ],
)
