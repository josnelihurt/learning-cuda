#pragma once

#include <memory>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wmissing-requires"
#include <grpcpp/grpcpp.h>
#pragma GCC diagnostic pop

#include "proto/_virtual_imports/image_processor_service_proto/image_processor_service.grpc.pb.h"

#include "cpp_accelerator/ports/grpc/processor_engine_provider.h"

namespace jrb::ports::grpc_service {

class ImageProcessorServiceImpl final : public cuda_learning::ImageProcessorService::Service {
 public:
  explicit ImageProcessorServiceImpl(std::shared_ptr<ProcessorEngineProvider> engine);
  ~ImageProcessorServiceImpl() override = default;

  ::grpc::Status ProcessImage(::grpc::ServerContext* context,
                              const cuda_learning::ProcessImageRequest* request,
                              cuda_learning::ProcessImageResponse* response) override;

  ::grpc::Status ListFilters(::grpc::ServerContext* context,
                             const cuda_learning::ListFiltersRequest* request,
                             cuda_learning::ListFiltersResponse* response) override;

  ::grpc::Status StreamProcessVideo(
      ::grpc::ServerContext* context,
      ::grpc::ServerReaderWriter<cuda_learning::ProcessImageResponse,
                                 cuda_learning::ProcessImageRequest>* stream) override;

 private:
  bool EnsureEngine() const;
  void CopyTraceContext(const cuda_learning::TraceContext& source,
                        cuda_learning::TraceContext* target) const;
  void CopyProcessMetadata(const cuda_learning::ProcessImageRequest& request,
                           cuda_learning::ProcessImageResponse* response) const;
  ::grpc::Status EngineFailureStatus(const cuda_learning::ProcessImageResponse& response) const;
  void PopulateListFiltersResponse(cuda_learning::ListFiltersResponse* response) const;

  std::shared_ptr<ProcessorEngineProvider> engine_;
};

}  // namespace jrb::ports::grpc_service


