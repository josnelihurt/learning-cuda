cc_library(
    name = "command_interface",
    hdrs = ["command_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core:result",
    ],
)

cc_library(
    name = "simple_kernel_command",
    srcs = ["simple_kernel_command.cpp"],
    hdrs = ["simple_kernel_command.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":command_interface",
        "//core:logger",
        "//lib:lib",
        "@spdlog//:spdlog",
    ],
)

cc_library(
    name = "grayscale_processor_command",
    srcs = ["grayscale_processor_command.cpp"],
    hdrs = ["grayscale_processor_command.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":command_interface",
        "//core:logger",
        "//lib:lib",
        "@spdlog//:spdlog",
    ],
)

cc_library(
    name = "command_factory",
    srcs = ["command_factory.cpp"],
    hdrs = ["command_factory.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":command_interface",
        ":simple_kernel_command",
        ":grayscale_processor_command",
        "//infrastructure/config:config",
    ],
)

