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
    srcs = [
        "grayscale_processor_command.cpp",
        "simple_kernel_command.cpp",
    ],
    hdrs = [
        "grayscale_processor_command.h",
        "simple_kernel_command.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":command_interface",
        "//cpp_accelerator/core:logger",
        "//cpp_accelerator/domain/interfaces",
        "//cpp_accelerator/domain/interfaces:processors",
        "@spdlog",
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
        "//cpp_accelerator/infrastructure/cuda:cuda_processors",
        "//cpp_accelerator/infrastructure/image:image_adapters",
    ],
)

cc_test(
    name = "commands_test",
    srcs = [
        "command_factory_test.cpp",
        "grayscale_processor_command_test.cpp",
        "simple_kernel_command_test.cpp",
    ],
    data = [
        "//data:lena.png",
    ],
    deps = [
        ":command_factory",
        ":command_interface",
        ":commands",
        "//cpp_accelerator/core:result",
        "//cpp_accelerator/domain/interfaces",
        "//cpp_accelerator/domain/interfaces:processors",
        "//cpp_accelerator/infrastructure/config",
        "@googletest//:gtest_main",
    ],
)
