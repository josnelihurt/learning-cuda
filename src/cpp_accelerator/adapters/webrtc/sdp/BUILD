load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "sdp_utils",
    srcs = ["sdp_utils.cpp"],
    hdrs = ["sdp_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_paullouisageneau_libdatachannel//:libdatachannel",
        "@spdlog",
    ],
)
