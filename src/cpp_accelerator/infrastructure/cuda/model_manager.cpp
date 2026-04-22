#include "src/cpp_accelerator/infrastructure/cuda/model_manager.h"
#include <spdlog/spdlog.h>

namespace jrb::infrastructure::cuda {

ModelManager& ModelManager::GetInstance() {
  static ModelManager instance;
  return instance;
}

void ModelManager::Initialize(const ModelRegistry& registry) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (initialized_) {
    spdlog::warn("ModelManager already initialized");
    return;
  }

  auto models = registry.GetAllModelInfo();
  for (const auto& model : models) {
    model_paths_[model.id] = model.model_path;
    spdlog::info("Registered model: {} ({}) at {}", model.id, model.name, model.model_path);
  }

  initialized_ = true;
  spdlog::info("ModelManager initialized with {} models", model_paths_.size());
}

std::shared_ptr<YOLODetector> ModelManager::GetDetector(
    const std::string& model_id, float confidence_threshold) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = model_paths_.find(model_id);
  if (it == model_paths_.end()) {
    spdlog::error("Model not found: {}", model_id);
    return nullptr;
  }

  try {
    auto detector = std::make_shared<YOLODetector>(it->second, confidence_threshold);
    spdlog::debug("Created detector for model: {} with confidence: {}", model_id, confidence_threshold);
    return detector;
  } catch (const std::exception& e) {
    spdlog::error("Failed to create detector for model {}: {}", model_id, e.what());
    return nullptr;
  }
}

std::vector<std::string> ModelManager::GetAvailableModels() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<std::string> model_ids;
  model_ids.reserve(model_paths_.size());
  for (const auto& [id, _] : model_paths_) {
    model_ids.push_back(id);
  }
  return model_ids;
}

}  // namespace jrb::infrastructure::cuda
