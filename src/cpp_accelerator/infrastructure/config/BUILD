cc_library(
    name = "config",
    srcs = ["config_manager.cpp"],
    hdrs = [
        "config_manager.h",
        "models/program_config.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//cpp_accelerator/core:result",
        "@lyra",
        "@spdlog",
    ],
)

cc_test(
    name = "config_test",
    srcs = ["config_manager_test.cpp"],
    deps = [
        ":config",
        "//cpp_accelerator/core:result",
        "@googletest//:gtest_main",
    ],
)
