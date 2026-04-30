#include "src/cpp_accelerator/application/commands/command_factory.h"
#include <gtest/gtest.h>
#include "src/cpp_accelerator/adapters/config/models/program_config.h"

namespace jrb::application::commands {
namespace {

class CommandFactoryTest : public ::testing::Test {
protected:
  CommandFactory uut;
};

TEST_F(CommandFactoryTest, FactoryIsConstructed) {
  EXPECT_NO_THROW(CommandFactory factory);
}

TEST_F(CommandFactoryTest, PassthroughTypeIsNotRegistered) {
  auto program_type = adapters::config::models::ProgramType::Passthrough;
  bool is_registered = uut.is_registered(program_type);
  EXPECT_FALSE(is_registered);
}

TEST_F(CommandFactoryTest, CudaImageFiltersTypeIsNotRegistered) {
  auto program_type = adapters::config::models::ProgramType::CudaImageFilters;
  bool is_registered = uut.is_registered(program_type);
  EXPECT_FALSE(is_registered);
}

TEST_F(CommandFactoryTest, CpuImageFiltersTypeIsNotRegistered) {
  auto program_type = adapters::config::models::ProgramType::CpuImageFilters;
  bool is_registered = uut.is_registered(program_type);
  EXPECT_FALSE(is_registered);
}

TEST_F(CommandFactoryTest, CreatePassthroughCommandReturnsNull) {
  adapters::config::models::ProgramConfig config{
      .input_image_path = "data/static_images/lena.png",
      .output_image_path = "data/output.png",
      .program_type = adapters::config::models::ProgramType::Passthrough};

  auto command = uut.create(config.program_type, config);
  ASSERT_EQ(command, nullptr);
}

TEST_F(CommandFactoryTest, CreateCudaImageFiltersCommandReturnsNull) {
  adapters::config::models::ProgramConfig config{
      .input_image_path = "data/static_images/lena.png",
      .output_image_path = "/tmp/test_grayscale_output.png",
      .program_type = adapters::config::models::ProgramType::CudaImageFilters};

  auto command = uut.create(config.program_type, config);
  ASSERT_EQ(command, nullptr);
}

}  // namespace
}  // namespace jrb::application::commands
