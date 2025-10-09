cc_library(
    name = "interfaces",
    hdrs = [
        "image_source.h",
        "image_sink.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "processors",
    hdrs = ["processors/i_image_processor.h"],
    visibility = ["//visibility:public"],
    deps = [":interfaces"],
)

