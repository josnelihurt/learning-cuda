#include "cpp_accelerator/core/logger.h"
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

namespace jrb::core {
namespace {

class LoggerTest : public ::testing::Test {
protected:
    void TearDown() override {
        // Reset logger after each test
        spdlog::drop_all();
        spdlog::set_default_logger(nullptr);
    }
};

TEST_F(LoggerTest, InitializeLoggerCreatesDefaultLogger) {
    // Arrange
    auto uut = initialize_logger;

    // Act
    uut();

    // Assert
    auto default_logger = spdlog::default_logger();
    ASSERT_NE(default_logger, nullptr);
    EXPECT_EQ(default_logger->name(), "main");
}

TEST_F(LoggerTest, InitializeLoggerSetsInfoLevel) {
    // Arrange
    auto uut = initialize_logger;

    // Act
    uut();

    // Assert
    auto default_logger = spdlog::default_logger();
    ASSERT_NE(default_logger, nullptr);
    EXPECT_EQ(spdlog::get_level(), spdlog::level::info);
}

TEST_F(LoggerTest, LoggerCanBeUsedAfterInitialization) {
    // Arrange
    auto uut = initialize_logger;
    uut();

    // Act & Assert (should not throw)
    EXPECT_NO_THROW(spdlog::info("Test log message"));
    EXPECT_NO_THROW(spdlog::debug("Debug message"));
    EXPECT_NO_THROW(spdlog::error("Error message"));
}

}  // namespace
}  // namespace jrb::core
