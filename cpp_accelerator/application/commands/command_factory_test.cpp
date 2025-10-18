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
  // Arrange & Act & Assert
  EXPECT_NO_THROW(CommandFactory factory);
}

TEST_F(CommandFactoryTest, SimpleTypeIsRegistered) {
  // Arrange
  auto program_type = infrastructure::config::models::ProgramType::Passthrough;

  // Act
  bool is_registered = uut.is_registered(program_type);

  // Assert
  EXPECT_TRUE(is_registered);
}

TEST_F(CommandFactoryTest, GrayscaleTypeIsRegistered) {
  // Arrange
  auto program_type = infrastructure::config::models::ProgramType::CudaImageFilters;

  // Act
  bool is_registered = uut.is_registered(program_type);

  // Assert
  EXPECT_TRUE(is_registered);
}

TEST_F(CommandFactoryTest, CreateSimpleCommand) {
  // Arrange
  infrastructure::config::models::ProgramConfig config{
      .input_image_path = "data/static_images/lena.png",
      .output_image_path = "data/output.png",
      .program_type = infrastructure::config::models::ProgramType::Passthrough};

  // Act
  auto command = uut.create(config.program_type, config);

  // Assert
  ASSERT_NE(command, nullptr);
  EXPECT_NO_THROW(command->execute());
}

TEST_F(CommandFactoryTest, CreateGrayscaleCommand) {
  // Arrange
  infrastructure::config::models::ProgramConfig config{
      .input_image_path = "data/static_images/lena.png",
      .output_image_path = "/tmp/test_grayscale_output.png",
      .program_type = infrastructure::config::models::ProgramType::CudaImageFilters};

  // Act
  auto command = uut.create(config.program_type, config);

  // Assert
  ASSERT_NE(command, nullptr);
}

TEST_F(CommandFactoryTest, CreateCommandUsesConfig) {
  // Arrange
  infrastructure::config::models::ProgramConfig config{
      .input_image_path = "data/static_images/lena.png",
      .output_image_path = "/tmp/custom_output.png",
      .program_type = infrastructure::config::models::ProgramType::CudaImageFilters};

  // Act
  auto command = uut.create(config.program_type, config);

  // Assert - Command should be created successfully with custom paths
  ASSERT_NE(command, nullptr);
}

TEST_F(CommandFactoryTest, CreateWithSimpleTypeReturnsValidCommand) {
  // Arrange
  infrastructure::config::models::ProgramConfig config{
      .input_image_path = "",
      .output_image_path = "",
      .program_type = infrastructure::config::models::ProgramType::Passthrough};

  // Act
  auto command = uut.create(infrastructure::config::models::ProgramType::Passthrough, config);

  // Assert
  ASSERT_NE(command, nullptr);
  auto result = command->execute();
  EXPECT_TRUE(result.success);
}

TEST_F(CommandFactoryTest, CreateWithGrayscaleTypeReturnsValidCommand) {
  // Arrange
  infrastructure::config::models::ProgramConfig config{
      .input_image_path = "data/static_images/lena.png",
      .output_image_path = "/tmp/factory_test_output.png",
      .program_type = infrastructure::config::models::ProgramType::CudaImageFilters};

  // Act
  auto command = uut.create(infrastructure::config::models::ProgramType::CudaImageFilters, config);

  // Assert
  ASSERT_NE(command, nullptr);
}

}  // namespace
}  // namespace jrb::application::commands
