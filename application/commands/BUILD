cc_library(
    name = "command_interface",
    hdrs = ["command_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core:result",
    ],
)

cc_library(
    name = "commands",
    srcs = [
        "simple_kernel_command.cpp",
        "grayscale_processor_command.cpp",
    ],
    hdrs = [
        "simple_kernel_command.h",
        "grayscale_processor_command.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":command_interface",
        "//domain/interfaces:interfaces",
        "//domain/interfaces:processors",
        "//core:logger",
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
        ":commands",
        "//infrastructure/cuda:cuda_processors",
        "//infrastructure/image:image_adapters",
        "//infrastructure/config:config",
    ],
)

