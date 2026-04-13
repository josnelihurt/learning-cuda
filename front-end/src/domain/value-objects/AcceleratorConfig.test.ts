import { describe, it, expect } from 'vitest';
import { AcceleratorConfig } from './AcceleratorConfig';
import { AcceleratorType } from '../../gen/common_pb';

describe('AcceleratorConfig', () => {
  // Test data builders
  const makeCPUAccelerator = () => new AcceleratorConfig('cpu');
  const makeGPUAccelerator = () => new AcceleratorConfig('gpu');
  const makeCudaAccelerator = () => new AcceleratorConfig('cuda');
  const makeInvalidType = () => 'opencl' as any;

  describe('Success Cases', () => {
    it('Success_CPUAccelerator', () => {
      // Arrange / Act
      const sut = makeCPUAccelerator();

      // Assert
      expect(sut.getType()).toBe('cpu');
      expect(sut.isCPU()).toBe(true);
      expect(sut.isGPU()).toBe(false);
    });

    it('Success_GPUAccelerator', () => {
      // Arrange / Act
      const sut = makeGPUAccelerator();

      // Assert
      expect(sut.getType()).toBe('gpu');
      expect(sut.isCPU()).toBe(false);
      expect(sut.isGPU()).toBe(true);
    });

    it('Success_NormalizeCudaToGpu', () => {
      // Arrange / Act
      const sut = makeCudaAccelerator();

      // Assert
      expect(sut.getType()).toBe('gpu');
      expect(sut.isGPU()).toBe(true);
      expect(sut.isCPU()).toBe(false);
    });

    it('Success_ProtocolConversionCPU', () => {
      // Arrange / Act
      const sut = makeCPUAccelerator();
      const protocol = sut.toProtocol();

      // Assert
      expect(protocol).toBe(AcceleratorType.CPU);
    });

    it('Success_ProtocolConversionGPU', () => {
      // Arrange / Act
      const sut = makeGPUAccelerator();
      const protocol = sut.toProtocol();

      // Assert
      expect(protocol).toBe(AcceleratorType.CUDA);
    });

    it('Success_ProtocolConversionCuda', () => {
      // Arrange / Act
      const sut = makeCudaAccelerator();
      const protocol = sut.toProtocol();

      // Assert
      expect(protocol).toBe(AcceleratorType.CUDA);
    });

    it('Success_EqualityCheck', () => {
      // Arrange
      const cpu1 = makeCPUAccelerator();
      const cpu2 = makeCPUAccelerator();
      const gpu = makeGPUAccelerator();

      // Assert
      expect(cpu1.equals(cpu2)).toBe(true);
      expect(cpu1.equals(gpu)).toBe(false);
    });

    it('Success_TypeCheckers', () => {
      // Arrange
      const cpuAccelerator = makeCPUAccelerator();
      const gpuAccelerator = makeGPUAccelerator();

      // Assert
      expect(cpuAccelerator.isCPU()).toBe(true);
      expect(cpuAccelerator.isGPU()).toBe(false);
      expect(gpuAccelerator.isCPU()).toBe(false);
      expect(gpuAccelerator.isGPU()).toBe(true);
    });

    it('Success_ToString', () => {
      // Arrange
      const cpuAccelerator = makeCPUAccelerator();
      const gpuAccelerator = makeGPUAccelerator();

      // Assert
      expect(cpuAccelerator.toString()).toBe('cpu');
      expect(gpuAccelerator.toString()).toBe('gpu');
    });
  });

  describe('Error Cases', () => {
    it('Error_EmptyType', () => {
      // Arrange / Act / Assert
      expect(() => new AcceleratorConfig(''))
        .toThrow('Accelerator type cannot be empty');
    });

    it('Error_NullType', () => {
      // Arrange / Act / Assert
      expect(() => new AcceleratorConfig(null as any))
        .toThrow('Accelerator type cannot be empty');
    });

    it('Error_InvalidType', () => {
      // Arrange / Act / Assert
      expect(() => new AcceleratorConfig(makeInvalidType()))
        .toThrow('Invalid accelerator type: opencl');
    });
  });

  describe('Edge Cases', () => {
    it('Edge_WhitespaceInType', () => {
      // Arrange / Act
      const sut = new AcceleratorConfig('  cpu  ');

      // Assert
      expect(sut.getType()).toBe('cpu');
      expect(sut.isCPU()).toBe(true);
    });

    it('Edge_CaseInsensitive', () => {
      // Arrange / Act
      const cpuUpper = new AcceleratorConfig('CPU');
      const gpuUpper = new AcceleratorConfig('GPU');

      // Assert
      expect(cpuUpper.getType()).toBe('cpu');
      expect(gpuUpper.getType()).toBe('gpu');
    });

    it('Edge_CudaVariant', () => {
      // Arrange / Act
      const sut = new AcceleratorConfig('cuda');

      // Assert
      expect(sut.getType()).toBe('gpu');
      expect(sut.isGPU()).toBe(true);
      expect(sut.toProtocol()).toBe(AcceleratorType.CUDA);
    });

    it('Edge_MixedCaseCuda', () => {
      // Arrange / Act
      const sut = new AcceleratorConfig('CUDA');

      // Assert
      expect(sut.getType()).toBe('gpu');
      expect(sut.isGPU()).toBe(true);
    });
  });

  describe('Table Driven Tests', () => {
    const testCases = [
      {
        name: 'Success_CPUAccelerator',
        input: 'cpu',
        expectedType: 'cpu',
        expectedProtocol: AcceleratorType.CPU,
        shouldThrow: false
      },
      {
        name: 'Success_GPUAccelerator',
        input: 'gpu',
        expectedType: 'gpu',
        expectedProtocol: AcceleratorType.CUDA,
        shouldThrow: false
      },
      {
        name: 'Success_CudaNormalizedToGpu',
        input: 'cuda',
        expectedType: 'gpu',
        expectedProtocol: AcceleratorType.CUDA,
        shouldThrow: false
      },
      {
        name: 'Success_CPUCaseInsensitive',
        input: 'CPU',
        expectedType: 'cpu',
        expectedProtocol: AcceleratorType.CPU,
        shouldThrow: false
      },
      {
        name: 'Success_GPUCaseInsensitive',
        input: 'GPU',
        expectedType: 'gpu',
        expectedProtocol: AcceleratorType.CUDA,
        shouldThrow: false
      },
      {
        name: 'Error_InvalidTypeOpenCL',
        input: 'opencl',
        expectedType: '',
        expectedProtocol: AcceleratorType.CPU,
        shouldThrow: true,
        expectedError: 'Invalid accelerator type'
      },
      {
        name: 'Error_EmptyType',
        input: '',
        expectedType: '',
        expectedProtocol: AcceleratorType.CPU,
        shouldThrow: true,
        expectedError: 'Accelerator type cannot be empty'
      }
    ];

    testCases.forEach(({ name, input, expectedType, expectedProtocol, shouldThrow, expectedError }) => {
      it(name, () => {
        if (shouldThrow) {
          // Arrange / Act / Assert
          expect(() => new AcceleratorConfig(input))
            .toThrow(expectedError);
        } else {
          // Arrange
          const sut = new AcceleratorConfig(input);

          // Assert
          expect(sut.getType()).toBe(expectedType);
          expect(sut.toProtocol()).toBe(expectedProtocol);
        }
      });
    });
  });
});
