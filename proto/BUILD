load("@rules_buf//buf:defs.bzl", "buf_lint_test")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

# Proto definitions
proto_library(
    name = "common_proto",
    srcs = ["common.proto"],
    strip_import_prefix = "/proto",
    visibility = ["//visibility:public"],
)

proto_library(
    name = "image_processor_service_proto",
    srcs = ["image_processor_service.proto"],
    strip_import_prefix = "/proto",
    visibility = ["//visibility:public"],
    deps = [":common_proto"],
)

proto_library(
    name = "config_service_proto",
    srcs = ["config_service.proto"],
    strip_import_prefix = "/proto",
    visibility = ["//visibility:public"],
    deps = [":common_proto"],
)

# C++ bindings
cc_proto_library(
    name = "common_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":common_proto"],
)

cc_proto_library(
    name = "image_processor_service_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":image_processor_service_proto"],
)

cc_proto_library(
    name = "config_service_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":config_service_proto"],
)

# Buf lint tests
buf_lint_test(
    name = "common_proto_lint",
    targets = [":common_proto"],
)

buf_lint_test(
    name = "image_processor_service_proto_lint",
    targets = [":image_processor_service_proto"],
)

buf_lint_test(
    name = "config_service_proto_lint",
    targets = [":config_service_proto"],
)

# Generate Connect-RPC using existing generated files
# Run manually: docker run --rm -v $(pwd):/workspace -u $(id -u):$(id -g) cuda-learning-bufgen:latest generate
# Go code is generated with buf and compiled with standard Go toolchain, not Bazel
