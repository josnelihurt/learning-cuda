#pragma once

#include "src/cpp_accelerator/domain/interfaces/i_yolo_detector.h"
#include "src/cpp_accelerator/adapters/compute/cuda/tensorrt/model_registry.h"
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <unordered_map>

namespace jrb::adapters::compute::cuda {

class ModelManager {
 public:
  static ModelManager& GetInstance();

  void Initialize(const ModelRegistry& registry);
  std::shared_ptr<IYoloDetector> GetDetector(const std::string& model_id, float confidence_threshold);
  std::vector<std::string> GetAvailableModels() const;

 private:
  ModelManager() = default;
  ~ModelManager() = default;
  ModelManager(const ModelManager&) = delete;
  ModelManager& operator=(const ModelManager&) = delete;

  std::unordered_map<std::string, std::string> model_paths_;
  mutable std::mutex mutex_;
  bool initialized_ = false;
};

}  // namespace jrb::adapters::compute::cuda
