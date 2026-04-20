#include "src/cpp_accelerator/infrastructure/cuda/model_registry.h"

namespace jrb::infrastructure::cuda {

void ModelRegistry::RegisterModel(const ModelInfo& info) {
  models_[info.id] = info;
}

const ModelInfo* ModelRegistry::GetModelInfo(const std::string& model_id) const {
  auto it = models_.find(model_id);
  if (it != models_.end()) {
    return &it->second;
  }
  return nullptr;
}

std::vector<std::string> ModelRegistry::GetAvailableModels() const {
  std::vector<std::string> model_ids;
  model_ids.reserve(models_.size());
  for (const auto& [id, _] : models_) {
    model_ids.push_back(id);
  }
  return model_ids;
}

std::vector<ModelInfo> ModelRegistry::GetAllModelInfo() const {
  std::vector<ModelInfo> models;
  models.reserve(models_.size());
  for (const auto& [_, info] : models_) {
    models.push_back(info);
  }
  return models;
}

}  // namespace jrb::infrastructure::cuda
