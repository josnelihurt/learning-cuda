cc_library(
    name = "gstreamer",
    copts = [
        "-I/usr/include/gstreamer-1.0",
        "-I/usr/include/glib-2.0",
    ],
    linkopts = [
        "-lgstreamer-1.0",
        "-lgstapp-1.0",
        "-lgobject-2.0",
        "-lglib-2.0",
    ],
    deps = ["//third_party/glib-config:glibconfig"],
    visibility = ["//visibility:public"],
)
