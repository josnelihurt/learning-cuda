import { describe, it, expect } from 'vitest';
import { GrayscaleAlgorithm } from './GrayscaleAlgorithm';
import { GrayscaleType } from '../../gen/common_pb';

describe('GrayscaleAlgorithm', () => {
  // Test data builders
  const makeBT601Algorithm = () => new GrayscaleAlgorithm('bt601');
  const makeBT709Algorithm = () => new GrayscaleAlgorithm('bt709');
  const makeAverageAlgorithm = () => new GrayscaleAlgorithm('average');
  const makeLightnessAlgorithm = () => new GrayscaleAlgorithm('lightness');
  const makeLuminosityAlgorithm = () => new GrayscaleAlgorithm('luminosity');
  const makeInvalidType = () => 'hsv' as any;

  describe('Success Cases', () => {
    it('Success_BT601Algorithm', () => {
      // Arrange / Act
      const sut = makeBT601Algorithm();

      // Assert
      expect(sut.getType()).toBe('bt601');
      expect(sut.isBT601()).toBe(true);
      expect(sut.isBT709()).toBe(false);
      expect(sut.isAverage()).toBe(false);
      expect(sut.isLightness()).toBe(false);
      expect(sut.isLuminosity()).toBe(false);
    });

    it('Success_BT709Algorithm', () => {
      // Arrange / Act
      const sut = makeBT709Algorithm();

      // Assert
      expect(sut.getType()).toBe('bt709');
      expect(sut.isBT601()).toBe(false);
      expect(sut.isBT709()).toBe(true);
      expect(sut.isAverage()).toBe(false);
      expect(sut.isLightness()).toBe(false);
      expect(sut.isLuminosity()).toBe(false);
    });

    it('Success_AverageAlgorithm', () => {
      // Arrange / Act
      const sut = makeAverageAlgorithm();

      // Assert
      expect(sut.getType()).toBe('average');
      expect(sut.isBT601()).toBe(false);
      expect(sut.isBT709()).toBe(false);
      expect(sut.isAverage()).toBe(true);
      expect(sut.isLightness()).toBe(false);
      expect(sut.isLuminosity()).toBe(false);
    });

    it('Success_LightnessAlgorithm', () => {
      // Arrange / Act
      const sut = makeLightnessAlgorithm();

      // Assert
      expect(sut.getType()).toBe('lightness');
      expect(sut.isBT601()).toBe(false);
      expect(sut.isBT709()).toBe(false);
      expect(sut.isAverage()).toBe(false);
      expect(sut.isLightness()).toBe(true);
      expect(sut.isLuminosity()).toBe(false);
    });

    it('Success_LuminosityAlgorithm', () => {
      // Arrange / Act
      const sut = makeLuminosityAlgorithm();

      // Assert
      expect(sut.getType()).toBe('luminosity');
      expect(sut.isBT601()).toBe(false);
      expect(sut.isBT709()).toBe(false);
      expect(sut.isAverage()).toBe(false);
      expect(sut.isLightness()).toBe(false);
      expect(sut.isLuminosity()).toBe(true);
    });

    it('Success_ProtocolConversion', () => {
      // Arrange / Act / Assert
      expect(makeBT601Algorithm().toProtocol()).toBe(GrayscaleType.BT601);
      expect(makeBT709Algorithm().toProtocol()).toBe(GrayscaleType.BT709);
      expect(makeAverageAlgorithm().toProtocol()).toBe(GrayscaleType.AVERAGE);
      expect(makeLightnessAlgorithm().toProtocol()).toBe(GrayscaleType.LIGHTNESS);
      expect(makeLuminosityAlgorithm().toProtocol()).toBe(GrayscaleType.LUMINOSITY);
    });

    it('Success_EqualityCheck', () => {
      // Arrange
      const bt601_1 = makeBT601Algorithm();
      const bt601_2 = makeBT601Algorithm();
      const bt709 = makeBT709Algorithm();

      // Assert
      expect(bt601_1.equals(bt601_2)).toBe(true);
      expect(bt601_1.equals(bt709)).toBe(false);
    });

    it('Success_TypeCheckers', () => {
      // Arrange
      const bt601 = makeBT601Algorithm();
      const bt709 = makeBT709Algorithm();
      const average = makeAverageAlgorithm();
      const lightness = makeLightnessAlgorithm();
      const luminosity = makeLuminosityAlgorithm();

      // Assert
      expect(bt601.isBT601()).toBe(true);
      expect(bt709.isBT709()).toBe(true);
      expect(average.isAverage()).toBe(true);
      expect(lightness.isLightness()).toBe(true);
      expect(luminosity.isLuminosity()).toBe(true);
    });

    it('Success_GetDescription', () => {
      // Arrange / Act / Assert
      expect(makeBT601Algorithm().getDescription()).toContain('ITU-R BT.601');
      expect(makeBT709Algorithm().getDescription()).toContain('ITU-R BT.709');
      expect(makeAverageAlgorithm().getDescription()).toContain('Average');
      expect(makeLightnessAlgorithm().getDescription()).toContain('Lightness');
      expect(makeLuminosityAlgorithm().getDescription()).toContain('Luminosity');
    });

    it('Success_ToString', () => {
      // Arrange / Act / Assert
      expect(makeBT601Algorithm().toString()).toBe('bt601');
      expect(makeBT709Algorithm().toString()).toBe('bt709');
      expect(makeAverageAlgorithm().toString()).toBe('average');
      expect(makeLightnessAlgorithm().toString()).toBe('lightness');
      expect(makeLuminosityAlgorithm().toString()).toBe('luminosity');
    });
  });

  describe('Error Cases', () => {
    it('Error_EmptyType', () => {
      // Arrange / Act / Assert
      expect(() => new GrayscaleAlgorithm(''))
        .toThrow('Grayscale algorithm type cannot be empty');
    });

    it('Error_NullType', () => {
      // Arrange / Act / Assert
      expect(() => new GrayscaleAlgorithm(null as any))
        .toThrow('Grayscale algorithm type cannot be empty');
    });

    it('Error_InvalidType', () => {
      // Arrange / Act / Assert
      expect(() => new GrayscaleAlgorithm(makeInvalidType()))
        .toThrow('Invalid grayscale algorithm type: hsv');
    });
  });

  describe('Edge Cases', () => {
    it('Edge_WhitespaceInType', () => {
      // Arrange / Act
      const sut = new GrayscaleAlgorithm('  bt601  ');

      // Assert
      expect(sut.getType()).toBe('bt601');
      expect(sut.isBT601()).toBe(true);
    });

    it('Edge_CaseVariations', () => {
      // Arrange / Act
      const bt601Upper = new GrayscaleAlgorithm('BT601');
      const bt709Mixed = new GrayscaleAlgorithm('Bt709');

      // Assert
      expect(bt601Upper.getType()).toBe('bt601');
      expect(bt709Mixed.getType()).toBe('bt709');
    });
  });

  describe('Table Driven Tests', () => {
    const testCases = [
      {
        name: 'Success_BT601Algorithm',
        input: 'bt601',
        expectedType: 'bt601',
        expectedProtocol: GrayscaleType.BT601,
        shouldThrow: false
      },
      {
        name: 'Success_BT709Algorithm',
        input: 'bt709',
        expectedType: 'bt709',
        expectedProtocol: GrayscaleType.BT709,
        shouldThrow: false
      },
      {
        name: 'Success_AverageAlgorithm',
        input: 'average',
        expectedType: 'average',
        expectedProtocol: GrayscaleType.AVERAGE,
        shouldThrow: false
      },
      {
        name: 'Success_LightnessAlgorithm',
        input: 'lightness',
        expectedType: 'lightness',
        expectedProtocol: GrayscaleType.LIGHTNESS,
        shouldThrow: false
      },
      {
        name: 'Success_LuminosityAlgorithm',
        input: 'luminosity',
        expectedType: 'luminosity',
        expectedProtocol: GrayscaleType.LUMINOSITY,
        shouldThrow: false
      },
      {
        name: 'Success_BT601CaseInsensitive',
        input: 'BT601',
        expectedType: 'bt601',
        expectedProtocol: GrayscaleType.BT601,
        shouldThrow: false
      },
      {
        name: 'Success_BT709CaseInsensitive',
        input: 'BT709',
        expectedType: 'bt709',
        expectedProtocol: GrayscaleType.BT709,
        shouldThrow: false
      },
      {
        name: 'Error_InvalidTypeHSV',
        input: 'hsv',
        expectedType: '',
        expectedProtocol: GrayscaleType.BT601,
        shouldThrow: true,
        expectedError: 'Invalid grayscale algorithm type'
      },
      {
        name: 'Error_EmptyType',
        input: '',
        expectedType: '',
        expectedProtocol: GrayscaleType.BT601,
        shouldThrow: true,
        expectedError: 'Grayscale algorithm type cannot be empty'
      }
    ];

    testCases.forEach(({ name, input, expectedType, expectedProtocol, shouldThrow, expectedError }) => {
      it(name, () => {
        if (shouldThrow) {
          // Arrange / Act / Assert
          expect(() => new GrayscaleAlgorithm(input))
            .toThrow(expectedError);
        } else {
          // Arrange
          const sut = new GrayscaleAlgorithm(input);

          // Assert
          expect(sut.getType()).toBe(expectedType);
          expect(sut.toProtocol()).toBe(expectedProtocol);
        }
      });
    });
  });
});
