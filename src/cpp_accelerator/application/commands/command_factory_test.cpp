#include "cpp_accelerator/application/commands/command_factory.h"
#include <gtest/gtest.h>
#include "cpp_accelerator/infrastructure/config/models/program_config.h"

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
  auto program_type = infrastructure::config::models::ProgramType::Passthrough;
  bool is_registered = uut.is_registered(program_type);
  EXPECT_FALSE(is_registered);
}

TEST_F(CommandFactoryTest, CudaImageFiltersTypeIsNotRegistered) {
  auto program_type = infrastructure::config::models::ProgramType::CudaImageFilters;
  bool is_registered = uut.is_registered(program_type);
  EXPECT_FALSE(is_registered);
}

TEST_F(CommandFactoryTest, CpuImageFiltersTypeIsNotRegistered) {
  auto program_type = infrastructure::config::models::ProgramType::CpuImageFilters;
  bool is_registered = uut.is_registered(program_type);
  EXPECT_FALSE(is_registered);
}

TEST_F(CommandFactoryTest, CreatePassthroughCommandReturnsNull) {
  infrastructure::config::models::ProgramConfig config{
      .input_image_path = "data/static_images/lena.png",
      .output_image_path = "data/output.png",
      .program_type = infrastructure::config::models::ProgramType::Passthrough};

  auto command = uut.create(config.program_type, config);
  ASSERT_EQ(command, nullptr);
}

TEST_F(CommandFactoryTest, CreateCudaImageFiltersCommandReturnsNull) {
  infrastructure::config::models::ProgramConfig config{
      .input_image_path = "data/static_images/lena.png",
      .output_image_path = "/tmp/test_grayscale_output.png",
      .program_type = infrastructure::config::models::ProgramType::CudaImageFilters};

  auto command = uut.create(config.program_type, config);
  ASSERT_EQ(command, nullptr);
}

}  // namespace
}  // namespace jrb::application::commands
