// ============================================================================
// TEMPORARY VALIDATION EXAMPLE - DELETE ONCE YOLO FULLY INTEGRATED
// ============================================================================
// This file validates that ONNX Runtime can initialize and YOLO model can load.
// Once YOLO is fully integrated into the filter pipeline, this file should be
// deleted along with its BUILD target.
// ============================================================================

#include "src/cpp_accelerator/infrastructure/cuda/i_yolo_detector.h"
#include "src/cpp_accelerator/infrastructure/cuda/yolo_factory.h"
#include "src/cpp_accelerator/infrastructure/image/image_loader.h"
#include "src/cpp_accelerator/domain/interfaces/image_buffer.h"
#include "spdlog/spdlog.h"

int main() {
    spdlog::info("YOLO Validation Example - Starting...");

    try {
        // Initialize detector with YOLO model
        // Note: Model file must be exported separately using Ultralytics
        constexpr const char* model_path = "data/models/yolov10n.onnx";

        spdlog::info("Loading YOLO model from: {}", model_path);
        auto detector = jrb::infrastructure::cuda::CreateYoloDetector(model_path, 0.5f);

        spdlog::info("YOLO model loaded successfully!");
        spdlog::info("CUDA Execution Provider: Active");

        // Load test image
        constexpr const char* image_path = "data/static_images/lena.png";
        jrb::infrastructure::image::ImageLoader image_loader(image_path);

        if (!image_loader.is_loaded()) {
            spdlog::error("Failed to load test image: {}", image_path);
            return 1;
        }

        spdlog::info("Test image loaded: {}x{}",
            image_loader.width(), image_loader.height());

        // Run detection
        spdlog::info("Running object detection...");

        jrb::domain::interfaces::ImageBuffer input(
            image_loader.data(),
            image_loader.width(),
            image_loader.height(),
            image_loader.channels()
        );

        jrb::domain::interfaces::ImageBufferMut output(
            nullptr,
            image_loader.width(),
            image_loader.height(),
            image_loader.channels()
        );

        jrb::domain::interfaces::FilterContext context(
            image_loader.data(),
            nullptr,
            image_loader.width(),
            image_loader.height(),
            image_loader.channels()
        );

        if (!detector->Apply(context)) {
            spdlog::error("Detection failed!");
            return 1;
        }

        // Report results
        const auto& detections = detector->GetDetections();
        spdlog::info("Detection complete! Found {} objects:", detections.size());

        for (const auto& det : detections) {
            spdlog::info("  - Class {}: {:.2f} confidence at ({:.1f}, {:.1f}, {:.1f}x{:.1f})",
                det.class_id, det.confidence, det.x, det.y, det.width, det.height);
        }

        spdlog::info("✓ YOLO validation PASSED - Library initialized successfully!");
        spdlog::info("  This file (yolo_validation_example.cpp) can now be deleted.");
        spdlog::info("  Remove the ':yolo_validation_example' target from BUILD file.");

        return 0;

    } catch (const std::exception& e) {
        spdlog::error("YOLO validation FAILED: {}", e.what());
        return 1;
    }
}
