load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "libdatachannel",
    srcs = glob([
        "src/*.cpp",
        "src/impl/*.cpp",
    ]),
    hdrs = glob([
        "include/rtc/**/*.h",
        "include/rtc/**/*.hpp",
        "src/impl/*.hpp",
    ]),
    includes = [
        "include",
        "include/rtc",
        "src",
    ],
    visibility = ["//visibility:public"],
    copts = [
        "-w",
        "-std=c++20",
        "-DRTC_ENABLE_MEDIA=1",
    ],
    deps = [
        "@com_github_sctplab_usrsctp//:usrsctp",
        "@com_github_paullouisageneau_libjuice//:libjuice",
        "@com_github_sergius02_plog//:plog",
        "@com_github_cisco_libsrtp//:srtp2",
        "@boringssl//:ssl",
        "@boringssl//:crypto",
    ],
    linkstatic = 1,
)

