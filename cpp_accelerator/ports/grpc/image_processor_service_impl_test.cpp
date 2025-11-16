#include "cpp_accelerator/ports/grpc/image_processor_service_impl.h"

#include <memory>
#include <string>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <grpcpp/create_channel.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/channel_arguments.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include "absl/strings/str_cat.h"

#include "cpp_accelerator/ports/grpc/processor_engine_provider.h"

namespace jrb::ports::grpc_service {

namespace {

class FakeEngineProvider : public ProcessorEngineProvider {
 public:
  bool process_should_succeed = true;
  std::string last_error_message = "failure";
  int processed_requests = 0;

  FakeEngineProvider() {
    auto* caps = capabilities_response_.mutable_capabilities();
    caps->set_api_version("1.0.0");
    caps->set_library_version("test");

    auto* filter = caps->add_filters();
    filter->set_id("grayscale");
    filter->set_name("Grayscale");
    filter->add_supported_accelerators(cuda_learning::ACCELERATOR_TYPE_CUDA);
    auto* param = filter->add_parameters();
    param->set_id("algorithm");
    param->set_name("Algorithm");
    param->set_type("select");
    param->add_options("bt601");
    param->add_options("bt709");
    param->set_default_value("bt601");
  }

  bool ProcessImage(const cuda_learning::ProcessImageRequest& request,
                    cuda_learning::ProcessImageResponse* response) override {
    ++processed_requests;
    last_request_trace_id = request.trace_context().traceparent();
    response->set_api_version(request.api_version());
    response->mutable_trace_context()->CopyFrom(request.trace_context());

    if (!process_should_succeed) {
      response->set_code(5);
      response->set_message(last_error_message);
      return false;
    }

    response->set_code(0);
    response->set_message("ok");
    response->set_width(request.width());
    response->set_height(request.height());
    response->set_channels(request.channels());
    response->set_image_data(request.image_data());
    return true;
  }

  bool GetCapabilities(cuda_learning::GetCapabilitiesResponse* response) override {
    response->CopyFrom(capabilities_response_);
    return true;
  }

  std::string last_request_trace_id;

 private:
  cuda_learning::GetCapabilitiesResponse capabilities_response_;
};

class ImageProcessorGrpcServiceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    provider_ = std::make_shared<FakeEngineProvider>();
    service_ = std::make_unique<ImageProcessorServiceImpl>(provider_);

    grpc::ServerBuilder builder;
    builder.RegisterService(service_.get());
    std::string server_address("127.0.0.1:0");
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials(), &selected_port_);
    server_ = builder.BuildAndStart();
    ASSERT_NE(server_, nullptr);

    auto channel = grpc::CreateChannel(absl::StrCat("127.0.0.1:", selected_port_),
                                       grpc::InsecureChannelCredentials());
    stub_ = cuda_learning::ImageProcessorService::NewStub(channel);
  }

  void TearDown() override {
    if (server_) {
      server_->Shutdown();
      server_->Wait();
    }
  }

  std::shared_ptr<FakeEngineProvider> provider_;
  std::unique_ptr<ImageProcessorServiceImpl> service_;
  std::unique_ptr<cuda_learning::ImageProcessorService::Stub> stub_;
  std::unique_ptr<grpc::Server> server_;
  int selected_port_ = 0;
};

TEST_F(ImageProcessorGrpcServiceTest, ProcessImageSuccess) {
  grpc::ClientContext context;
  cuda_learning::ProcessImageRequest request;
  cuda_learning::ProcessImageResponse response;
  request.set_api_version("v1");
  request.mutable_trace_context()->set_traceparent("trace-123");
  request.set_width(4);
  request.set_height(4);
  request.set_channels(3);
  request.set_image_data("abcd");

  auto status = stub_->ProcessImage(&context, request, &response);

  ASSERT_TRUE(status.ok());
  EXPECT_EQ(response.code(), 0);
  EXPECT_EQ(response.image_data(), "abcd");
  EXPECT_EQ(provider_->processed_requests, 1);
  EXPECT_EQ(response.trace_context().traceparent(), "trace-123");
}

TEST_F(ImageProcessorGrpcServiceTest, ProcessImageFailurePropagatesStatus) {
  provider_->process_should_succeed = false;
  provider_->last_error_message = "no filters";

  grpc::ClientContext context;
  cuda_learning::ProcessImageRequest request;
  cuda_learning::ProcessImageResponse response;
  request.set_api_version("v1");

  auto status = stub_->ProcessImage(&context, request, &response);

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_message(), "no filters");
  EXPECT_EQ(provider_->processed_requests, 1);
}

TEST_F(ImageProcessorGrpcServiceTest, ListFiltersReturnsCapabilities) {
  grpc::ClientContext context;
  cuda_learning::ListFiltersRequest request;
  cuda_learning::ListFiltersResponse response;
  request.set_api_version("v1");
  request.mutable_trace_context()->set_traceparent("trace");

  auto status = stub_->ListFilters(&context, request, &response);

  ASSERT_TRUE(status.ok());
  ASSERT_EQ(response.filters_size(), 1);
  const auto& filter = response.filters(0);
  EXPECT_EQ(filter.id(), "grayscale");
  ASSERT_EQ(filter.parameters_size(), 1);
  EXPECT_EQ(filter.parameters(0).options_size(), 2);
}

TEST_F(ImageProcessorGrpcServiceTest, StreamProcessVideoHandlesMultipleFrames) {
  grpc::ClientContext context;
  auto stream = stub_->StreamProcessVideo(&context);
  ASSERT_NE(stream, nullptr);

  for (int i = 0; i < 3; ++i) {
    cuda_learning::ProcessImageRequest request;
    request.set_api_version("v1");
    request.set_width(2);
    request.set_height(2);
    request.set_channels(1);
    request.set_image_data("zz");
    stream->Write(request);
  }
  stream->WritesDone();

  cuda_learning::ProcessImageResponse response;
  int responses = 0;
  while (stream->Read(&response)) {
    EXPECT_EQ(response.code(), 0);
    ++responses;
  }
  auto status = stream->Finish();
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(responses, 3);
}

}  // namespace

}  // namespace jrb::ports::grpc_service


