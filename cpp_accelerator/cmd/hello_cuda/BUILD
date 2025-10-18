cc_binary(
    name = "hello_cuda",
    srcs = ["main.cpp"],
    data = ["//data:static_images/lena.png"],
    deps = [
        "//cpp_accelerator/application/commands:command_factory",
        "//cpp_accelerator/core:logger",
        "//cpp_accelerator/infrastructure/config",
        "@spdlog",
    ],
)
