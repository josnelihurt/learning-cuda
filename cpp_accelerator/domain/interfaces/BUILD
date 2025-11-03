cc_library(
    name = "interfaces",
    hdrs = [
        "i_pixel_getter.h",
        "image_buffer.h",
        "image_sink.h",
        "image_source.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "processors",
    hdrs = ["processors/i_image_processor.h"],
    visibility = ["//visibility:public"],
    deps = [":interfaces"],
)

