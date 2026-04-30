load("@rules_cc//cc:defs.bzl", "cc_library")

# Jetson (aarch64) uses Argus/nvarguscamerasrc; x86 with --config=v4l2-camera uses
# standard V4L2 + GStreamer; default x86 uses a stub (detection works, streaming logs a warning).
_JETSON_GST_COPTS = [
    "-I/usr/include/gstreamer-1.0",
    "-I/usr/include/glib-2.0",
    "-I/usr/lib/aarch64-linux-gnu/glib-2.0/include",
]
_X86_GST_COPTS = [
    "-I/usr/include/gstreamer-1.0",
    "-I/usr/include/glib-2.0",
    # glibconfig.h (arch-specific) is exposed via //third_party/glib-config:glibconfig
]

cc_library(
    name = "camera_detector",
    srcs = select({
        "@platforms//cpu:aarch64": ["camera_detector_jetson.cpp"],
        "//conditions:default": ["camera_detector_v4l2.cpp"],
    }),
    hdrs = ["camera_detector.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto:accelerator_control_cc_proto",
        "@spdlog",
    ] + select({
        "@platforms//cpu:aarch64": [],
        "//bazel/flags:v4l2_camera_enabled": ["//third_party/glib-config:glibconfig"],
        "//conditions:default": [],
    }),
    linkopts = select({
        "@platforms//cpu:aarch64": ["-lgstreamer-1.0", "-lgobject-2.0", "-lglib-2.0"],
        "//bazel/flags:v4l2_camera_enabled": ["-lgstreamer-1.0", "-lgobject-2.0", "-lglib-2.0"],
        "//conditions:default": [],
    }),
    copts = select({
        "@platforms//cpu:aarch64": _JETSON_GST_COPTS,
        "//bazel/flags:v4l2_camera_enabled": _X86_GST_COPTS,
        "//conditions:default": [],
    }),
)

cc_library(
    name = "gst_camera_source",
    srcs = select({
        "@platforms//cpu:aarch64": ["gst_camera_source_jetson.cpp"],
        "//bazel/flags:v4l2_camera_enabled": ["gst_camera_source_v4l2.cpp"],
        "//conditions:default": ["gst_camera_source_stub.cpp"],
    }),
    hdrs = ["gst_camera_source.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_paullouisageneau_libdatachannel//:libdatachannel",
        "@spdlog",
    ] + select({
        "@platforms//cpu:aarch64": [],
        "//bazel/flags:v4l2_camera_enabled": ["//third_party/glib-config:glibconfig"],
        "//conditions:default": [],
    }),
    linkopts = select({
        "@platforms//cpu:aarch64": ["-lgstreamer-1.0", "-lgstapp-1.0", "-lgobject-2.0", "-lglib-2.0"],
        "//bazel/flags:v4l2_camera_enabled": ["-lgstreamer-1.0", "-lgstapp-1.0", "-lgobject-2.0", "-lglib-2.0"],
        "//conditions:default": [],
    }),
    copts = select({
        "@platforms//cpu:aarch64": _JETSON_GST_COPTS,
        "//bazel/flags:v4l2_camera_enabled": _X86_GST_COPTS,
        "//conditions:default": [],
    }),
)
