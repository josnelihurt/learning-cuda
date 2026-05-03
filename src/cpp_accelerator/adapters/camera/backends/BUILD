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
    srcs = ["v4l2_backend.cpp"],
    hdrs = ["v4l2_backend.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":camera_backend",
        "@spdlog",
    ] + select({
        "//bazel/flags:v4l2_camera_enabled": [
            "//third_party/glib-config:glibconfig",
        ],
        "//conditions:default": [],
    }),
    copts = select({
        "//bazel/flags:v4l2_camera_enabled": [
            "-I/usr/include/gstreamer-1.0",
            "-I/usr/include/glib-2.0",
        ],
        "//conditions:default": [],
    }),
    linkopts = select({
        "//bazel/flags:v4l2_camera_enabled": [
            "-lgstreamer-1.0",
            "-lgstapp-1.0",
            "-lgobject-2.0",
            "-lglib-2.0",
        ],
        "//conditions:default": [],
    }),
)

# NVIDIA Argus backend (GStreamer nvarguscamerasrc)
cc_library(
    name = "nvidia_argus_backend",
    srcs = ["nvidia_argus_backend.cpp"],
    hdrs = ["nvidia_argus_backend.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":camera_backend",
        "//src/cpp_accelerator/adapters/camera:gpu_frame_processor",
        "@com_github_paullouisageneau_libdatachannel//:libdatachannel",
        "@spdlog",
    ] + select({
        "//bazel/flags:nvidia_argus_camera_enabled": [
            "//third_party/glib-config:glibconfig",
        ],
        "//conditions:default": [],
    }),
    copts = select({
        "//bazel/flags:nvidia_argus_camera_enabled": [
            "-I/usr/include/gstreamer-1.0",
            "-I/usr/include/glib-2.0",
        ],
        "//conditions:default": [],
    }),
    linkopts = select({
        "//bazel/flags:nvidia_argus_camera_enabled": [
            "-lgstreamer-1.0",
            "-lgstapp-1.0",
            "-lgobject-2.0",
            "-lglib-2.0",
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
