load("@rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "filesystem",
    srcs = ["static_image_repository.go"],
    importpath = "github.com/jrb/cuda-learning/webserver/pkg/infrastructure/filesystem",
    visibility = ["//visibility:public"],
    deps = [
        "//webserver/pkg/domain",
        "@io_opentelemetry_go_otel//:otel",
        "@io_opentelemetry_go_otel//attribute",
        "@io_opentelemetry_go_otel_trace//:trace",
    ],
)

go_test(
    name = "filesystem_test",
    srcs = ["static_image_repository_test.go"],
    embed = [":filesystem"],
    deps = [
        "@com_github_stretchr_testify//assert",
        "@com_github_stretchr_testify//require",
    ],
)

