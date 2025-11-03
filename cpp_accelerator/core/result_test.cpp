#include "cpp_accelerator/core/result.h"

#include <gtest/gtest.h>
#include <string>

namespace jrb::core {
namespace {

TEST(ResultTest, VoidResultOkCreatesSuccessResult) {
  // Arrange & Act
  Result<void> result = Result<void>::ok("Operation successful", 0);

  // Assert
  ASSERT_TRUE(result.success);
  EXPECT_EQ(result.message, "Operation successful");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_TRUE(static_cast<bool>(result));
}

TEST(ResultTest, VoidResultErrorCreatesFailureResult) {
  // Arrange & Act
  Result<void> result = Result<void>::error("Operation failed", 1);

  // Assert
  ASSERT_FALSE(result.success);
  EXPECT_EQ(result.message, "Operation failed");
  EXPECT_EQ(result.exit_code, 1);
  EXPECT_FALSE(static_cast<bool>(result));
}

TEST(ResultTest, VoidResultOkWithDefaultParameters) {
  // Arrange & Act
  Result<void> result = Result<void>::ok();

  // Assert
  ASSERT_TRUE(result.success);
  EXPECT_EQ(result.message, "");
  EXPECT_EQ(result.exit_code, 0);
}

TEST(ResultTest, TypedResultOkCreatesSuccessResultWithValue) {
  // Arrange & Act
  Result<int> result = Result<int>::ok(42, "Value retrieved", 0);

  // Assert
  ASSERT_TRUE(result.success);
  EXPECT_EQ(result.message, "Value retrieved");
  EXPECT_EQ(result.exit_code, 0);
  ASSERT_TRUE(result.value.has_value());
  EXPECT_EQ(result.value.value(), 42);
  EXPECT_TRUE(static_cast<bool>(result));
}

TEST(ResultTest, TypedResultErrorCreatesFailureResultWithoutValue) {
  // Arrange & Act
  Result<int> result = Result<int>::error("Failed to retrieve value", 1);

  // Assert
  ASSERT_FALSE(result.success);
  EXPECT_EQ(result.message, "Failed to retrieve value");
  EXPECT_EQ(result.exit_code, 1);
  EXPECT_FALSE(result.value.has_value());
  EXPECT_FALSE(static_cast<bool>(result));
}

TEST(ResultTest, TypedResultOkWithDefaultParameters) {
  // Arrange & Act
  Result<std::string> result = Result<std::string>::ok("test", "", 0);

  // Assert
  ASSERT_TRUE(result.success);
  EXPECT_EQ(result.message, "");
  EXPECT_EQ(result.exit_code, 0);
  ASSERT_TRUE(result.value.has_value());
  EXPECT_EQ(result.value.value(), "test");
}

TEST(ResultTest, TypedResultWithStringValue) {
  // Arrange & Act
  Result<std::string> result = Result<std::string>::ok("hello world", "String operation", 0);

  // Assert
  ASSERT_TRUE(result.success);
  EXPECT_EQ(result.message, "String operation");
  ASSERT_TRUE(result.value.has_value());
  EXPECT_EQ(result.value.value(), "hello world");
}

TEST(ResultTest, TypedResultWithMovedValue) {
  // Arrange
  std::string original = "original string";

  // Act
  Result<std::string> result = Result<std::string>::ok(std::move(original), "Moved value", 0);

  // Assert
  ASSERT_TRUE(result.success);
  ASSERT_TRUE(result.value.has_value());
  EXPECT_EQ(result.value.value(), "original string");
  EXPECT_TRUE(original.empty());
}

TEST(ResultTest, ResultBoolConversionWorksCorrectly) {
  // Arrange
  Result<void> success_result = Result<void>::ok();
  Result<void> failure_result = Result<void>::error("Error");

  // Act & Assert
  EXPECT_TRUE(static_cast<bool>(success_result));
  EXPECT_FALSE(static_cast<bool>(failure_result));
}

TEST(ResultTest, DifferentExitCodesArePreserved) {
  // Arrange & Act
  Result<void> result1 = Result<void>::error("Error 1", 1);
  Result<void> result2 = Result<void>::error("Error 2", 2);
  Result<void> result3 = Result<void>::ok("Success", 0);

  // Assert
  EXPECT_EQ(result1.exit_code, 1);
  EXPECT_EQ(result2.exit_code, 2);
  EXPECT_EQ(result3.exit_code, 0);
}

}  // namespace
}  // namespace jrb::core

