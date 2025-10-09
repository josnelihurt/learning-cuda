#include "cpp_accelerator/application/commands/simple_kernel_command.h"
#include "cpp_accelerator/domain/interfaces/processors/i_image_processor.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace jrb::application::commands {
namespace {

namespace mock {

class MockImageProcessor : public domain::interfaces::IImageProcessor {
public:
    MOCK_METHOD(bool, process, 
        (domain::interfaces::IImageSource& source,
         domain::interfaces::IImageSink& sink,
         const std::string& output_path), 
        (override));
};

}  // namespace mock

class SimpleKernelCommandTest : public ::testing::Test {
};

TEST_F(SimpleKernelCommandTest, ExecuteCallsProcessor) {
    // Arrange
    auto mock_processor = std::make_unique<mock::MockImageProcessor>();
    auto* mock_ptr = mock_processor.get();
    
    EXPECT_CALL(*mock_ptr, process(::testing::_, ::testing::_, ::testing::_))
        .Times(1)
        .WillOnce(::testing::Return(true));
    
    SimpleKernelCommand uut(std::move(mock_processor));

    // Act
    auto result = uut.execute();

    // Assert
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.exit_code, 0);
}

TEST_F(SimpleKernelCommandTest, ExecuteReturnsSuccessResult) {
    // Arrange
    auto mock_processor = std::make_unique<mock::MockImageProcessor>();
    auto* mock_ptr = mock_processor.get();
    
    EXPECT_CALL(*mock_ptr, process(::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));
    
    SimpleKernelCommand uut(std::move(mock_processor));

    // Act
    auto result = uut.execute();

    // Assert
    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.message.empty());
}

TEST_F(SimpleKernelCommandTest, ConstructorAcceptsProcessor) {
    // Arrange
    auto mock_processor = std::make_unique<mock::MockImageProcessor>();

    // Act & Assert (should not throw)
    EXPECT_NO_THROW(SimpleKernelCommand uut(std::move(mock_processor)));
}

}  // namespace
}  // namespace jrb::application::commands
