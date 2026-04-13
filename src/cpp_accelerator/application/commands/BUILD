cc_library(
    name = "command_interface",
    hdrs = ["command_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//cpp_accelerator/core:result",
    ],
)

cc_library(
    name = "commands",
    srcs = [],
    hdrs = [],
    visibility = ["//visibility:public"],
    deps = [
        ":command_interface",
    ],
)

cc_library(
    name = "command_factory",
    srcs = ["command_factory.cpp"],
    hdrs = ["command_factory.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":command_interface",
        ":commands",
        "//cpp_accelerator/infrastructure/config",
    ],
)

cc_test(
    name = "commands_test",
    srcs = [
        "command_factory_test.cpp",
    ],
    data = [
        "//data:static_images/lena.png",
    ],
    deps = [
        ":command_factory",
        ":command_interface",
        ":commands",
        "//cpp_accelerator/core:result",
        "//cpp_accelerator/infrastructure/config",
        "@googletest//:gtest_main",
    ],
)
