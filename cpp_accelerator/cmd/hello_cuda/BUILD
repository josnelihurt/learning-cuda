cc_binary(
    name = "hello_cuda",
    srcs = ["main.cpp"],
    deps = [
        "//cpp_accelerator/core:logger",
        "//cpp_accelerator/infrastructure/config:config",
        "//cpp_accelerator/application/commands:command_factory",
        "@spdlog//:spdlog",
    ],
    data = ["//data:lena.png"],
)
