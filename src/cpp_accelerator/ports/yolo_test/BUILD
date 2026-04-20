load("@rules_cc//cc:defs.bzl", "cc_binary")

# YOLO Validation Example
# This binary validates that ONNX Runtime can initialize and YOLO model can load.
# Once YOLO is fully integrated into the filter pipeline, this test serves as a
# standalone validation tool.
cc_binary(
    name = "yolo_validation_example",
    srcs = ["yolo_validation_example.cpp"],
    data = ["//data:static_images/lena.png"],
    deps = [
        "//src/cpp_accelerator/infrastructure/cuda:yolo_detector",
        "//src/cpp_accelerator/infrastructure/image:image_adapters",
        "@spdlog",
    ],
)
