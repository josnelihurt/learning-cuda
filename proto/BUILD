load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@rules_go//go:def.bzl", "go_library")
load("@rules_buf//buf:defs.bzl", "buf_lint_test")

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
    importpath = "github.com/jrb/cuda-learning/proto/gen",
    visibility = ["//visibility:public"],
    deps = [
        "@org_golang_google_protobuf//reflect/protoreflect",
        "@org_golang_google_protobuf//runtime/protoimpl",
    ],
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

