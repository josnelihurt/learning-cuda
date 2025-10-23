import { describe, it, expect } from 'vitest';
import { FilterData } from './FilterData';
import { FilterType } from '../../gen/common_pb';

describe('FilterData', () => {
  // Test data builders
  const makeNoneFilter = () => new FilterData('none');
  const makeGrayscaleFilter = () => new FilterData('grayscale');
  const makeFilterWithRadius = (radius: number) => new FilterData('none', { radius });
  const makeFilterWithIntensity = (intensity: number) => new FilterData('none', { intensity });
  const makeInvalidType = () => 'blur' as any;

  describe('Success Cases', () => {
    it('Success_NoneFilter', () => {
      // Arrange / Act
      const sut = makeNoneFilter();

      // Assert
      expect(sut.getType()).toBe('none');
      expect(sut.getParameters()).toEqual({});
      expect(sut.hasParameters()).toBe(false);
    });

    it('Success_GrayscaleFilter', () => {
      // Arrange / Act
      const sut = makeGrayscaleFilter();

      // Assert
      expect(sut.getType()).toBe('grayscale');
      expect(sut.isGrayscale()).toBe(true);
      expect(sut.isNone()).toBe(false);
    });

    it('Success_FilterWithValidParameters', () => {
      // Arrange / Act
      const sut = makeFilterWithRadius(50);

      // Assert
      expect(sut.hasParameters()).toBe(true);
      expect(sut.getParameter('radius')).toBe(50);
    });

    it('Success_ProtocolConversionNone', () => {
      // Arrange / Act
      const sut = makeNoneFilter();
      const protocol = sut.toProtocol();

      // Assert
      expect(protocol).toBe(FilterType.NONE);
    });

    it('Success_ProtocolConversionGrayscale', () => {
      // Arrange / Act
      const sut = makeGrayscaleFilter();
      const protocol = sut.toProtocol();

      // Assert
      expect(protocol).toBe(FilterType.GRAYSCALE);
    });

    it('Success_EqualityCheck', () => {
      // Arrange
      const filter1 = makeNoneFilter();
      const filter2 = makeNoneFilter();
      const filter3 = makeGrayscaleFilter();

      // Assert
      expect(filter1.equals(filter2)).toBe(true);
      expect(filter1.equals(filter3)).toBe(false);
    });

    it('Success_ImmutableParameters', () => {
      // Arrange
      const sut = makeFilterWithRadius(50);
      const params = sut.getParameters();

      // Act
      params.radius = 100;

      // Assert
      expect(sut.getParameter('radius')).toBe(50);
    });

    it('Success_TypeCheckers', () => {
      // Arrange
      const noneFilter = makeNoneFilter();
      const grayscaleFilter = makeGrayscaleFilter();

      // Assert
      expect(noneFilter.isNone()).toBe(true);
      expect(noneFilter.isGrayscale()).toBe(false);
      expect(grayscaleFilter.isNone()).toBe(false);
      expect(grayscaleFilter.isGrayscale()).toBe(true);
    });

    it('Success_ToString', () => {
      // Arrange
      const noneFilter = makeNoneFilter();
      const filterWithParams = makeFilterWithRadius(50);

      // Assert
      expect(noneFilter.toString()).toBe('none');
      expect(filterWithParams.toString()).toContain('none');
      expect(filterWithParams.toString()).toContain('radius');
    });
  });

  describe('Error Cases', () => {
    it('Error_EmptyFilterType', () => {
      // Arrange / Act / Assert
      expect(() => new FilterData(''))
        .toThrow('Filter type cannot be empty');
    });

    it('Error_NullFilterType', () => {
      // Arrange / Act / Assert
      expect(() => new FilterData(null as any))
        .toThrow('Filter type cannot be empty');
    });

    it('Error_InvalidFilterType', () => {
      // Arrange / Act / Assert
      expect(() => new FilterData(makeInvalidType()))
        .toThrow('Invalid filter type: blur');
    });

    it('Error_InvalidRadiusNegative', () => {
      // Arrange / Act / Assert
      expect(() => makeFilterWithRadius(-1))
        .toThrow('Radius must be a number between 0 and 100');
    });

    it('Error_InvalidRadiusTooBig', () => {
      // Arrange / Act / Assert
      expect(() => makeFilterWithRadius(101))
        .toThrow('Radius must be a number between 0 and 100');
    });

    it('Error_InvalidRadiusNotNumber', () => {
      // Arrange / Act / Assert
      expect(() => new FilterData('none', { radius: 'invalid' }))
        .toThrow('Radius must be a number between 0 and 100');
    });

    it('Error_InvalidIntensityNegative', () => {
      // Arrange / Act / Assert
      expect(() => makeFilterWithIntensity(-0.1))
        .toThrow('Intensity must be a number between 0 and 1');
    });

    it('Error_InvalidIntensityTooBig', () => {
      // Arrange / Act / Assert
      expect(() => makeFilterWithIntensity(1.1))
        .toThrow('Intensity must be a number between 0 and 1');
    });

    it('Error_InvalidIntensityNotNumber', () => {
      // Arrange / Act / Assert
      expect(() => new FilterData('none', { intensity: 'invalid' }))
        .toThrow('Intensity must be a number between 0 and 1');
    });
  });

  describe('Edge Cases', () => {
    it('Edge_EmptyParameters', () => {
      // Arrange / Act
      const sut = new FilterData('none', {});

      // Assert
      expect(sut.hasParameters()).toBe(false);
      expect(sut.getParameters()).toEqual({});
    });

    it('Edge_MultipleParameters', () => {
      // Arrange / Act
      const sut = new FilterData('none', { radius: 50, intensity: 0.5 });

      // Assert
      expect(sut.getParameter('radius')).toBe(50);
      expect(sut.getParameter('intensity')).toBe(0.5);
    });

    it('Edge_RadiusBoundaries', () => {
      // Arrange / Act / Assert
      expect(() => makeFilterWithRadius(0)).not.toThrow();
      expect(() => makeFilterWithRadius(100)).not.toThrow();
    });

    it('Edge_IntensityBoundaries', () => {
      // Arrange / Act / Assert
      expect(() => makeFilterWithIntensity(0)).not.toThrow();
      expect(() => makeFilterWithIntensity(1)).not.toThrow();
    });

    it('Edge_WhitespaceInType', () => {
      // Arrange / Act / Assert
      expect(() => new FilterData('  none  ' as any))
        .toThrow('Invalid filter type');
    });
  });

  describe('Table Driven Tests', () => {
    const testCases = [
      {
        name: 'Success_NoneFilterNoParams',
        type: 'none' as const,
        params: {},
        expectedProtocol: FilterType.NONE,
        shouldThrow: false
      },
      {
        name: 'Success_GrayscaleFilterNoParams',
        type: 'grayscale' as const,
        params: {},
        expectedProtocol: FilterType.GRAYSCALE,
        shouldThrow: false
      },
      {
        name: 'Error_InvalidTypeBlur',
        type: 'blur' as any,
        params: {},
        expectedProtocol: FilterType.NONE,
        shouldThrow: true,
        expectedError: 'Invalid filter type'
      },
      {
        name: 'Error_InvalidTypeEmpty',
        type: '' as any,
        params: {},
        expectedProtocol: FilterType.NONE,
        shouldThrow: true,
        expectedError: 'Filter type cannot be empty'
      }
    ];

    testCases.forEach(({ name, type, params, expectedProtocol, shouldThrow, expectedError }) => {
      it(name, () => {
        if (shouldThrow) {
          // Arrange / Act / Assert
          expect(() => new FilterData(type, params))
            .toThrow(expectedError);
        } else {
          // Arrange
          const sut = new FilterData(type, params);

          // Assert
          expect(sut.getType()).toBe(type);
          expect(sut.toProtocol()).toBe(expectedProtocol);
        }
      });
    });
  });
});
