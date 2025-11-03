cc_library(
    name = "result",
    hdrs = ["result.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "logger",
    srcs = ["logger.cpp"],
    hdrs = ["logger.h"],
    visibility = ["//visibility:public"],
    deps = [
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
