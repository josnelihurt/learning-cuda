import { describe, it, expect } from 'vitest';
import { ImageData } from './ImageData';

describe('ImageData', () => {
  // Test data builders
  const makeValidPngData = () => 'data:image/png;base64,iVBORw0KGgo=';
  const makeValidJpegData = () => 'data:image/jpeg;base64,/9j/4AAQ=';
  const makeValidGifData = () => 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
  const makeInvalidData = () => 'invalid-data';
  const makeEmptyData = () => '';
  const makeNullData = () => null as any;
  const makeUndefinedData = () => undefined as any;
  
  describe('Success Cases', () => {
    it('Success_ValidPngImage', () => {
      // Arrange
      const base64 = makeValidPngData();
      const width = 100;
      const height = 200;

      // Act
      const sut = new ImageData(base64, width, height);

      // Assert
      expect(sut.getBase64()).toBe(base64);
      expect(sut.getWidth()).toBe(width);
      expect(sut.getHeight()).toBe(height);
    });

    it('Success_ValidJpegImage', () => {
      // Arrange
      const base64 = makeValidJpegData();
      const width = 300;
      const height = 400;

      // Act
      const sut = new ImageData(base64, width, height);

      // Assert
      expect(sut.getBase64()).toBe(base64);
      expect(sut.getWidth()).toBe(width);
      expect(sut.getHeight()).toBe(height);
    });

    it('Success_ValidGifImage', () => {
      // Arrange
      const base64 = makeValidGifData();
      const width = 50;
      const height = 50;

      // Act
      const sut = new ImageData(base64, width, height);

      // Assert
      expect(sut.getBase64()).toBe(base64);
      expect(sut.getWidth()).toBe(width);
      expect(sut.getHeight()).toBe(height);
    });

    it('Success_AspectRatioCalculation', () => {
      // Arrange / Act
      const sut = new ImageData(makeValidPngData(), 100, 200);

      // Assert
      expect(sut.getAspectRatio()).toBe(0.5);
    });

    it('Success_SquareImage', () => {
      // Arrange / Act
      const sut = new ImageData(makeValidPngData(), 100, 100);

      // Assert
      expect(sut.getAspectRatio()).toBe(1);
    });

    it('Success_WideImage', () => {
      // Arrange / Act
      const sut = new ImageData(makeValidPngData(), 200, 100);

      // Assert
      expect(sut.getAspectRatio()).toBe(2);
    });

    it('Success_TallImage', () => {
      // Arrange / Act
      const sut = new ImageData(makeValidPngData(), 100, 300);

      // Assert
      expect(sut.getAspectRatio()).toBe(1/3);
    });

    it('Success_EqualityCheck', () => {
      // Arrange
      const base64 = makeValidPngData();
      const image1 = new ImageData(base64, 100, 200);
      const image2 = new ImageData(base64, 100, 200);
      const image3 = new ImageData(base64, 150, 200);

      // Assert
      expect(image1.equals(image2)).toBe(true);
      expect(image1.equals(image3)).toBe(false);
    });

    it('Success_ToDataUrl', () => {
      // Arrange
      const base64 = makeValidPngData();
      const sut = new ImageData(base64, 100, 200);

      // Act
      const dataUrl = sut.toDataUrl();

      // Assert
      expect(dataUrl).toBe(base64);
    });
  });

  describe('Error Cases', () => {
    it('Error_EmptyBase64', () => {
      // Arrange / Act / Assert
      expect(() => new ImageData(makeEmptyData(), 100, 200))
        .toThrow('Image data cannot be empty');
    });

    it('Error_NullBase64', () => {
      // Arrange / Act / Assert
      expect(() => new ImageData(makeNullData(), 100, 200))
        .toThrow('Image data cannot be empty');
    });

    it('Error_UndefinedBase64', () => {
      // Arrange / Act / Assert
      expect(() => new ImageData(makeUndefinedData(), 100, 200))
        .toThrow('Image data cannot be empty');
    });

    it('Error_InvalidFormat', () => {
      // Arrange / Act / Assert
      expect(() => new ImageData(makeInvalidData(), 100, 200))
        .toThrow('Invalid image format: must start with data:image/');
    });

    it('Error_NegativeWidth', () => {
      // Arrange / Act / Assert
      expect(() => new ImageData(makeValidPngData(), -1, 200))
        .toThrow('Image dimensions must be positive');
    });

    it('Error_NegativeHeight', () => {
      // Arrange / Act / Assert
      expect(() => new ImageData(makeValidPngData(), 100, -1))
        .toThrow('Image dimensions must be positive');
    });

    it('Error_ZeroWidth', () => {
      // Arrange / Act / Assert
      expect(() => new ImageData(makeValidPngData(), 0, 200))
        .toThrow('Image dimensions must be positive');
    });

    it('Error_ZeroHeight', () => {
      // Arrange / Act / Assert
      expect(() => new ImageData(makeValidPngData(), 100, 0))
        .toThrow('Image dimensions must be positive');
    });

    it('Error_FloatWidth', () => {
      // Arrange / Act / Assert
      expect(() => new ImageData(makeValidPngData(), 100.5, 200))
        .toThrow('Image dimensions must be integers');
    });

    it('Error_FloatHeight', () => {
      // Arrange / Act / Assert
      expect(() => new ImageData(makeValidPngData(), 100, 200.7))
        .toThrow('Image dimensions must be integers');
    });

    it('Error_BothNegativeDimensions', () => {
      // Arrange / Act / Assert
      expect(() => new ImageData(makeValidPngData(), -1, -1))
        .toThrow('Image dimensions must be positive');
    });

    it('Error_BothZeroDimensions', () => {
      // Arrange / Act / Assert
      expect(() => new ImageData(makeValidPngData(), 0, 0))
        .toThrow('Image dimensions must be positive');
    });
  });

  describe('Edge Cases', () => {
    it('Edge_MinimumDimensions', () => {
      // Arrange / Act
      const sut = new ImageData(makeValidPngData(), 1, 1);

      // Assert
      expect(sut.getWidth()).toBe(1);
      expect(sut.getHeight()).toBe(1);
      expect(sut.getAspectRatio()).toBe(1);
    });

    it('Edge_LargeDimensions', () => {
      // Arrange / Act
      const sut = new ImageData(makeValidPngData(), 4096, 4096);

      // Assert
      expect(sut.getWidth()).toBe(4096);
      expect(sut.getHeight()).toBe(4096);
      expect(sut.getAspectRatio()).toBe(1);
    });

    it('Edge_VeryLongBase64', () => {
      // Arrange
      const longBase64 = 'data:image/png;base64,' + 'A'.repeat(10000);
      
      // Act
      const sut = new ImageData(longBase64, 100, 200);

      // Assert
      expect(sut.getBase64()).toBe(longBase64);
      expect(sut.getWidth()).toBe(100);
      expect(sut.getHeight()).toBe(200);
    });

    it('Edge_WhitespaceBase64', () => {
      // Arrange
      const base64WithWhitespace = '  data:image/png;base64,test  ';
      
      // Act
      const sut = new ImageData(base64WithWhitespace, 100, 200);

      // Assert
      expect(sut.getBase64()).toBe(base64WithWhitespace);
    });

    it('Edge_MaximumIntegerDimensions', () => {
      // Arrange
      const maxInt = Number.MAX_SAFE_INTEGER;
      
      // Act
      const sut = new ImageData(makeValidPngData(), maxInt, maxInt);

      // Assert
      expect(sut.getWidth()).toBe(maxInt);
      expect(sut.getHeight()).toBe(maxInt);
    });
  });

  describe('Table Driven Tests', () => {
    const testCases = [
      {
        name: 'Success_ValidPng',
        base64: 'data:image/png;base64,test',
        width: 100,
        height: 200,
        expectedAspectRatio: 0.5,
        shouldThrow: false
      },
      {
        name: 'Success_ValidJpeg',
        base64: 'data:image/jpeg;base64,test',
        width: 300,
        height: 400,
        expectedAspectRatio: 0.75,
        shouldThrow: false
      },
      {
        name: 'Error_EmptyBase64',
        base64: '',
        width: 100,
        height: 200,
        expectedAspectRatio: 0,
        shouldThrow: true,
        expectedError: 'Image data cannot be empty'
      },
      {
        name: 'Error_InvalidFormat',
        base64: 'invalid-format',
        width: 100,
        height: 200,
        expectedAspectRatio: 0,
        shouldThrow: true,
        expectedError: 'Invalid image format'
      },
      {
        name: 'Error_NegativeWidth',
        base64: 'data:image/png;base64,test',
        width: -1,
        height: 200,
        expectedAspectRatio: 0,
        shouldThrow: true,
        expectedError: 'Image dimensions must be positive'
      }
    ];

    testCases.forEach(({ name, base64, width, height, expectedAspectRatio, shouldThrow, expectedError }) => {
      it(name, () => {
        if (shouldThrow) {
          // Arrange / Act / Assert
          expect(() => new ImageData(base64, width, height))
            .toThrow(expectedError);
        } else {
          // Arrange
          const sut = new ImageData(base64, width, height);

          // Assert
          expect(sut.getBase64()).toBe(base64);
          expect(sut.getWidth()).toBe(width);
          expect(sut.getHeight()).toBe(height);
          expect(sut.getAspectRatio()).toBe(expectedAspectRatio);
        }
      });
    });
  });
});
