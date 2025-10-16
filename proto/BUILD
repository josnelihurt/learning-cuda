load("@rules_buf//buf:defs.bzl", "buf_lint_test")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@rules_go//go:def.bzl", "go_library")
load("@rules_go//proto:def.bzl", "go_proto_library")
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

# Proto messages (package gen)
go_library(
    name = "image_processing_go_proto",
    srcs = glob(
        ["gen/*.pb.go"],
        allow_empty = True,
    ),
    embed = [":gen_go_proto"],
    importpath = "github.com/jrb/cuda-learning/proto/gen",
    visibility = ["//visibility:public"],
)

# Connect-RPC handlers (package genconnect)
go_library(
    name = "image_processing_connect_go",
    srcs = glob(
        ["gen/genconnect/*.go"],
        allow_empty = True,
    ),
    importpath = "github.com/jrb/cuda-learning/proto/gen/genconnect",
    visibility = ["//visibility:public"],
    deps = [
        ":image_processing_go_proto",
        "@com_connectrpc_connect//:connect",
    ],
)

go_proto_library(
    name = "gen_go_proto",
    compilers = ["@rules_go//proto:go_grpc"],
    importpath = "github.com/jrb/cuda-learning/proto/gen",
    protos = [
        ":common_proto",
        ":image_processor_service_proto",
        ":config_service_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [],
)
