#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace jrb::infrastructure::cuda {

struct ModelInfo {
  std::string id;
  std::string name;
  std::string model_path;
  std::string description;
};

class ModelRegistry {
 public:
  ModelRegistry() = default;

  void RegisterModel(const ModelInfo& info);
  const ModelInfo* GetModelInfo(const std::string& model_id) const;
  std::vector<std::string> GetAvailableModels() const;
  std::vector<ModelInfo> GetAllModelInfo() const;

 private:
  std::unordered_map<std::string, ModelInfo> models_;
};

}  // namespace jrb::infrastructure::cuda
