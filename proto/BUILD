load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@rules_go//proto:def.bzl", "go_proto_library")

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

# Go bindings
go_proto_library(
    name = "image_processing_go_proto",
    importpath = "github.com/jrb/cuda-learning/proto",
    proto = ":image_processing_proto",
    visibility = ["//visibility:public"],
)

