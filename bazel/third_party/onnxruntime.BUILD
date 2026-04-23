cc_library(
    name = "onnxruntime",
    srcs = ["lib/libonnxruntime.so"],
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
    linkopts = ["-lonnxruntime"],
)
