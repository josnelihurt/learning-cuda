load("@rules_buf//buf:defs.bzl", "buf_lint_test")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@rules_go//go:def.bzl", "go_library")
load("@rules_go//proto:def.bzl", "go_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

# Proto definitions
proto_library(
    name = "image_processing_proto",
    srcs = ["image_processing.proto"],
    visibility = ["//visibility:public"],
)

# C++ bindings
cc_proto_library(
    name = "image_processing_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":image_processing_proto"],
)

# Buf lint test
buf_lint_test(
    name = "image_processing_proto_lint",
    targets = [":image_processing_proto"],
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
    proto = ":image_processing_proto",
    visibility = ["//visibility:public"],
)
