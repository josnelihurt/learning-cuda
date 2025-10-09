cc_binary(
    name = "hello_cuda",
    srcs = ["main.cpp"],
    deps = [
        "//core:logger",
        "//infrastructure/config:config",
        "//application/commands:command_factory",
        "@spdlog//:spdlog",
    ],
    data = ["//data:lena.png"],
)
