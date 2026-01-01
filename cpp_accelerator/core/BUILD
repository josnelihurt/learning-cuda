cc_library(
    name = "result",
    hdrs = ["result.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "otel_log_sink",
    srcs = ["otel_log_sink.cpp"],
    hdrs = ["otel_log_sink.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@spdlog",
        "@io_opentelemetry_cpp//api:api",
        "@io_opentelemetry_cpp//exporters/otlp:otlp_http_log_record_exporter",
        "@io_opentelemetry_cpp//sdk/src/logs:logs",
        "@io_opentelemetry_cpp//sdk/src/resource:resource",
        "@curl//:curl",
        "//cpp_accelerator/ports/shared_lib:library_version",
    ],
)

cc_library(
    name = "logger",
    srcs = ["logger.cpp"],
    hdrs = ["logger.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":otel_log_sink",
        "@spdlog",
    ],
)

cc_library(
    name = "telemetry",
    srcs = ["telemetry_stub.cpp"],
    hdrs = ["telemetry.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@spdlog",
    ],
)

cc_test(
    name = "logger_test",
    srcs = ["logger_test.cpp"],
    deps = [
        ":logger",
        "@googletest//:gtest_main",
        "@spdlog",
    ],
)

cc_test(
    name = "result_test",
    srcs = ["result_test.cpp"],
    deps = [
        ":result",
        "@googletest//:gtest_main",
    ],
)
