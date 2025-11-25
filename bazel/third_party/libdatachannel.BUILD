load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "libdatachannel",
    srcs = glob([
        "src/**/*.cpp",
        "src/**/*.c",
    ]),
    hdrs = glob([
        "include/datachannel/**/*.h",
        "include/rtc/**/*.h",
    ]),
    includes = [
        "include",
    ],
    visibility = ["//visibility:public"],
    copts = ["-w"],
    deps = [
        "@com_github_sctplab_usrsctp//:usrsctp",
        "@com_github_paullouisageneau_libjuice//:libjuice",
    ],
    linkstatic = 1,
)

