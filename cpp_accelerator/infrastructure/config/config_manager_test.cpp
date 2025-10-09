#include "cpp_accelerator/infrastructure/config/config_manager.h"
#include "cpp_accelerator/infrastructure/config/models/program_config.h"
#include <gtest/gtest.h>
#include <vector>
#include <string>

namespace jrb::infrastructure::config {
namespace {

class ConfigManagerTest : public ::testing::Test {
protected:
    // Helper to convert string vector to span of char*
    std::span<const char*> make_args(const std::vector<std::string>& str_args) {
        char_ptrs_.clear();
        for (const auto& str : str_args) {
            char_ptrs_.push_back(str.c_str());
        }
        return std::span<const char*>(char_ptrs_.data(), char_ptrs_.size());
    }

private:
    std::vector<const char*> char_ptrs_;
};

TEST_F(ConfigManagerTest, ParseWithDefaultValues) {
    // Arrange
    std::vector<std::string> args = {"program"};
    auto args_span = make_args(args);

    // Act
    auto uut_result = ConfigManager::parse(args_span);

    // Assert
    ASSERT_TRUE(uut_result.success);
    ASSERT_TRUE(uut_result.value.has_value());
    EXPECT_EQ(uut_result.value->input_image_path, "data/lena.png");
    EXPECT_EQ(uut_result.value->output_image_path, "data/output.png");
    EXPECT_EQ(uut_result.value->program_type, models::ProgramType::Grayscale);
}

TEST_F(ConfigManagerTest, ParseWithCustomInputPath) {
    // Arrange
    std::vector<std::string> args = {"program", "-i", "custom_input.png"};
    auto args_span = make_args(args);

    // Act
    auto uut_result = ConfigManager::parse(args_span);

    // Assert
    ASSERT_TRUE(uut_result.success);
    ASSERT_TRUE(uut_result.value.has_value());
    EXPECT_EQ(uut_result.value->input_image_path, "custom_input.png");
    EXPECT_EQ(uut_result.value->output_image_path, "data/output.png");
}

TEST_F(ConfigManagerTest, ParseWithCustomOutputPath) {
    // Arrange
    std::vector<std::string> args = {"program", "-o", "custom_output.png"};
    auto args_span = make_args(args);

    // Act
    auto uut_result = ConfigManager::parse(args_span);

    // Assert
    ASSERT_TRUE(uut_result.success);
    ASSERT_TRUE(uut_result.value.has_value());
    EXPECT_EQ(uut_result.value->input_image_path, "data/lena.png");
    EXPECT_EQ(uut_result.value->output_image_path, "custom_output.png");
}

TEST_F(ConfigManagerTest, ParseWithSimpleProgramType) {
    // Arrange
    std::vector<std::string> args = {"program", "-t", "simple"};
    auto args_span = make_args(args);

    // Act
    auto uut_result = ConfigManager::parse(args_span);

    // Assert
    ASSERT_TRUE(uut_result.success);
    ASSERT_TRUE(uut_result.value.has_value());
    EXPECT_EQ(uut_result.value->program_type, models::ProgramType::Simple);
}

TEST_F(ConfigManagerTest, ParseWithGrayscaleProgramType) {
    // Arrange
    std::vector<std::string> args = {"program", "-t", "grayscale"};
    auto args_span = make_args(args);

    // Act
    auto uut_result = ConfigManager::parse(args_span);

    // Assert
    ASSERT_TRUE(uut_result.success);
    ASSERT_TRUE(uut_result.value.has_value());
    EXPECT_EQ(uut_result.value->program_type, models::ProgramType::Grayscale);
}

TEST_F(ConfigManagerTest, ParseWithInvalidProgramType) {
    // Arrange
    std::vector<std::string> args = {"program", "-t", "invalid"};
    auto args_span = make_args(args);

    // Act
    auto uut_result = ConfigManager::parse(args_span);

    // Assert
    EXPECT_FALSE(uut_result.success);
    EXPECT_FALSE(uut_result.value.has_value());
    EXPECT_NE(uut_result.message.find("Invalid program type"), std::string::npos);
}

TEST_F(ConfigManagerTest, ParseWithAllCustomParameters) {
    // Arrange
    std::vector<std::string> args = {
        "program",
        "-i", "my_input.png",
        "-o", "my_output.png",
        "-t", "simple"
    };
    auto args_span = make_args(args);

    // Act
    auto uut_result = ConfigManager::parse(args_span);

    // Assert
    ASSERT_TRUE(uut_result.success);
    ASSERT_TRUE(uut_result.value.has_value());
    EXPECT_EQ(uut_result.value->input_image_path, "my_input.png");
    EXPECT_EQ(uut_result.value->output_image_path, "my_output.png");
    EXPECT_EQ(uut_result.value->program_type, models::ProgramType::Simple);
}

TEST_F(ConfigManagerTest, ParseWithHelpFlag) {
    // Arrange
    std::vector<std::string> args = {"program", "-h"};
    auto args_span = make_args(args);

    // Act
    auto uut_result = ConfigManager::parse(args_span);

    // Assert
    EXPECT_FALSE(uut_result.success);
    EXPECT_FALSE(uut_result.value.has_value());
    EXPECT_EQ(uut_result.exit_code, 0);
    EXPECT_NE(uut_result.message.find("Help requested"), std::string::npos);
}

TEST_F(ConfigManagerTest, ParseWithLongOptionNames) {
    // Arrange
    std::vector<std::string> args = {
        "program",
        "--input", "long_input.png",
        "--output", "long_output.png",
        "--type", "simple"
    };
    auto args_span = make_args(args);

    // Act
    auto uut_result = ConfigManager::parse(args_span);

    // Assert
    ASSERT_TRUE(uut_result.success);
    ASSERT_TRUE(uut_result.value.has_value());
    EXPECT_EQ(uut_result.value->input_image_path, "long_input.png");
    EXPECT_EQ(uut_result.value->output_image_path, "long_output.png");
    EXPECT_EQ(uut_result.value->program_type, models::ProgramType::Simple);
}

}  // namespace
}  // namespace jrb::infrastructure::config
