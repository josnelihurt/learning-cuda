load("@rules_cc//cc:defs.bzl", "cc_library")

# Unified camera implementations using modular backends (V4L2, NVIDIA Argus, Stub).
# Each backend can be independently enabled via Bazel flags:
# - --//bazel/flags:v4l2_camera=true    (USB cameras on x86 or Jetson)
# - --//bazel/flags:nvidia_argus_camera=true (NVIDIA Argus on Jetson)
# If no backends are enabled, the stub backend is used (returns no cameras).

cc_library(
    name = "camera_detector",
    srcs = [
        "camera_detector.cpp",
        "camera_detector_impl.cpp",
        "camera_detector_impl.h",
    ],
    hdrs = ["camera_detector.h"],
    visibility = ["//visibility:public"],
    defines = select({
        "//bazel/flags:v4l2_camera_enabled": ["CAMERA_BACKEND_V4L2_ENABLED"],
        "//conditions:default": [],
    }) + select({
        "//bazel/flags:nvidia_argus_camera_enabled": ["CAMERA_BACKEND_NVIDIA_ARGUS_ENABLED"],
        "//conditions:default": [],
    }),
    deps = [
        ":backends",
        "//proto:accelerator_control_cc_proto",
        "@spdlog",
    ],
)

cc_library(
    name = "gst_camera_source",
    srcs = [
        "gst_camera_source.cpp",
        "gst_camera_source_impl.cpp",
        "gst_camera_source_impl.h",
    ],
    hdrs = ["gst_camera_source.h"],
    visibility = ["//visibility:public"],
    defines = select({
        "//bazel/flags:v4l2_camera_enabled": ["CAMERA_BACKEND_V4L2_ENABLED"],
        "//conditions:default": [],
    }) + select({
        "//bazel/flags:nvidia_argus_camera_enabled": ["CAMERA_BACKEND_NVIDIA_ARGUS_ENABLED"],
        "//conditions:default": [],
    }),
    deps = [
        ":backends",
        "@com_github_paullouisageneau_libdatachannel//:libdatachannel",
        "@spdlog",
    ],
)

cc_library(
    name = "backends",
    deps = [
        "//src/cpp_accelerator/adapters/camera/backends:camera_backend",
        "//src/cpp_accelerator/adapters/camera/backends:stub_backend",
    ] + select({
        "//bazel/flags:v4l2_camera_enabled": ["//src/cpp_accelerator/adapters/camera/backends:v4l2_backend"],
        "//conditions:default": [],
    }) + select({
        "//bazel/flags:nvidia_argus_camera_enabled": ["//src/cpp_accelerator/adapters/camera/backends:nvidia_argus_backend"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "camera_hub",
    srcs = ["camera_hub.cpp"],
    hdrs = ["camera_hub.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gst_camera_source",
        "@com_github_paullouisageneau_libdatachannel//:libdatachannel",
        "@spdlog",
    ],
)
