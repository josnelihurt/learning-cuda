cc_binary(
    name = "hello_cuda",
    srcs = ["main.cpp"],
    deps = [
        "//core:logger",
        "//infrastructure/config:config",
        "//app:command_factory",
        "@spdlog//:spdlog",
    ],
    data = ["//data:lena.png"],
)
