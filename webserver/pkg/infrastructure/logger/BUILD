load("@rules_go//go:def.bzl", "go_library")

go_library(
    name = "logger",
    srcs = ["logger.go"],
    importpath = "github.com/jrb/cuda-learning/webserver/pkg/infrastructure/logger",
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_rs_zerolog//:zerolog",
        "@io_opentelemetry_go_otel_trace//:trace",
    ],
)

