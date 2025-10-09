cc_library(
    name = "config",
    srcs = ["config_manager.cpp"],
    hdrs = [
        "config_manager.h",
        "models/program_config.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//core:result",
        "@lyra//:lyra",
        "@spdlog//:spdlog",
    ],
)

cc_test(
    name = "config_test",
    srcs = ["config_manager_test.cpp"],
    deps = [
        ":config",
        "//core:result",
        "@googletest//:gtest_main",
    ],
)
