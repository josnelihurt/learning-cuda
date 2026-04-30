load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

# Accelerator capability flags.
# Usage: bazel build --//bazel/flags:cuda=true ...
# Or via named .bazelrc configs: bazel build --config=cuda ...
#
# CPU is always enabled. All other accelerators are opt-in.

bool_flag(
    name = "cuda",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

bool_flag(
    name = "opencl",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

bool_flag(
    name = "vulkan",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

bool_flag(
    name = "v4l2_camera",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "cuda_enabled",
    flag_values = {":cuda": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "opencl_enabled",
    flag_values = {":opencl": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "vulkan_enabled",
    flag_values = {":vulkan": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "v4l2_camera_enabled",
    flag_values = {":v4l2_camera": "true"},
    visibility = ["//visibility:public"],
)
