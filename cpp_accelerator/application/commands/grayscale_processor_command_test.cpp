#include "cpp_accelerator/application/commands/grayscale_processor_command.h"
#include "cpp_accelerator/domain/interfaces/processors/i_image_processor.h"
#include "cpp_accelerator/domain/interfaces/image_source.h"
#include "cpp_accelerator/domain/interfaces/image_sink.h"
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

class MockImageSource : public domain::interfaces::IImageSource {
public:
    MOCK_METHOD(int, width, (), (const, override));
    MOCK_METHOD(int, height, (), (const, override));
    MOCK_METHOD(int, channels, (), (const, override));
    MOCK_METHOD(const unsigned char*, data, (), (const, override));
    MOCK_METHOD(bool, is_valid, (), (const, override));
};

class MockImageSink : public domain::interfaces::IImageSink {
public:
    MOCK_METHOD(bool, write, 
        (const char* filepath,
         const unsigned char* data,
         int width,
         int height,
         int channels), 
        (override));
};

}  // namespace mock

class GrayscaleProcessorCommandTest : public ::testing::Test {
};

TEST_F(GrayscaleProcessorCommandTest, ExecuteWithValidSource) {
    // Arrange
    auto mock_processor = std::make_unique<mock::MockImageProcessor>();
    auto mock_source = std::make_unique<mock::MockImageSource>();
    auto mock_sink = std::make_unique<mock::MockImageSink>();
    
    auto* processor_ptr = mock_processor.get();
    auto* source_ptr = mock_source.get();
    
    EXPECT_CALL(*source_ptr, is_valid())
        .WillOnce(::testing::Return(true));
    
    EXPECT_CALL(*processor_ptr, process(::testing::_, ::testing::_, "output.png"))
        .Times(1)
        .WillOnce(::testing::Return(true));
    
    GrayscaleProcessorCommand uut(
        std::move(mock_processor),
        std::move(mock_source),
        std::move(mock_sink),
        "output.png"
    );

    // Act
    auto result = uut.execute();

    // Assert
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.exit_code, 0);
}

TEST_F(GrayscaleProcessorCommandTest, ExecuteWithInvalidSource) {
    // Arrange
    auto mock_processor = std::make_unique<mock::MockImageProcessor>();
    auto mock_source = std::make_unique<mock::MockImageSource>();
    auto mock_sink = std::make_unique<mock::MockImageSink>();
    
    auto* source_ptr = mock_source.get();
    
    EXPECT_CALL(*source_ptr, is_valid())
        .WillOnce(::testing::Return(false));
    
    GrayscaleProcessorCommand uut(
        std::move(mock_processor),
        std::move(mock_source),
        std::move(mock_sink),
        "output.png"
    );

    // Act
    auto result = uut.execute();

    // Assert
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.exit_code, 1);
    EXPECT_NE(result.message.find("Failed to load"), std::string::npos);
}

TEST_F(GrayscaleProcessorCommandTest, ExecuteWhenProcessorFails) {
    // Arrange
    auto mock_processor = std::make_unique<mock::MockImageProcessor>();
    auto mock_source = std::make_unique<mock::MockImageSource>();
    auto mock_sink = std::make_unique<mock::MockImageSink>();
    
    auto* processor_ptr = mock_processor.get();
    auto* source_ptr = mock_source.get();
    
    EXPECT_CALL(*source_ptr, is_valid())
        .WillOnce(::testing::Return(true));
    
    EXPECT_CALL(*processor_ptr, process(::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));
    
    GrayscaleProcessorCommand uut(
        std::move(mock_processor),
        std::move(mock_source),
        std::move(mock_sink),
        "output.png"
    );

    // Act
    auto result = uut.execute();

    // Assert
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.exit_code, 1);
    EXPECT_NE(result.message.find("processing failed"), std::string::npos);
}

TEST_F(GrayscaleProcessorCommandTest, ConstructorAcceptsDependencies) {
    // Arrange
    auto mock_processor = std::make_unique<mock::MockImageProcessor>();
    auto mock_source = std::make_unique<mock::MockImageSource>();
    auto mock_sink = std::make_unique<mock::MockImageSink>();

    // Act & Assert (should not throw)
    EXPECT_NO_THROW(
        GrayscaleProcessorCommand uut(
            std::move(mock_processor),
            std::move(mock_source),
            std::move(mock_sink),
            "test.png"
        )
    );
}

}  // namespace
}  // namespace jrb::application::commands
