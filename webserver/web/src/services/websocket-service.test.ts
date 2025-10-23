import { describe, it, expect, vi, beforeEach } from 'vitest';
import { WebSocketService } from './websocket-service';
import { ImageData, FilterData, AcceleratorConfig, GrayscaleAlgorithm } from '../domain/value-objects';
import { AcceleratorType } from '../gen/common_pb';

describe('WebSocketService - ImageData Integration', () => {
  const makeValidImageData = () => new ImageData(
    'data:image/png;base64,test',
    100,
    200
  );

  const makeMockStatsManager = () => ({
    updateWebSocketStatus: vi.fn(),
  });

  const makeMockCameraManager = () => ({
    setProcessing: vi.fn(),
  });

  const makeMockToastManager = () => ({
    warning: vi.fn(),
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Success Cases', () => {
    it('Success_SendFrameWithImageData', () => {
      // Arrange
      const mockWs = {
        readyState: 1, // WebSocket.OPEN
        send: vi.fn(),
      };
      const statsManager = makeMockStatsManager();
      const cameraManager = makeMockCameraManager();
      const toastManager = makeMockToastManager();
      
      const sut = new WebSocketService(statsManager, cameraManager, toastManager);
      (sut as any).ws = mockWs;

      const image = makeValidImageData();

      // Act
      sut.sendFrameWithImageData(image, ['grayscale'], 'gpu', 'bt709');

      // Assert
      expect(mockWs.send).toHaveBeenCalledOnce();
      expect(cameraManager.setProcessing).toHaveBeenCalledWith(true);
    });

    it('Success_SendFrameUsesImageDataInternally', () => {
      // Arrange
      const mockWs = {
        readyState: 1, // WebSocket.OPEN
        send: vi.fn(),
      };
      const statsManager = makeMockStatsManager();
      const cameraManager = makeMockCameraManager();
      const toastManager = makeMockToastManager();
      
      const sut = new WebSocketService(statsManager, cameraManager, toastManager);
      (sut as any).ws = mockWs;

      // Act
      sut.sendFrame('data:image/png;base64,test', 100, 200, ['grayscale'], 'gpu', 'bt709');

      // Assert
      expect(mockWs.send).toHaveBeenCalledOnce();
      expect(cameraManager.setProcessing).toHaveBeenCalledWith(true);
    });

    it('Success_ValidImageDataWithDifferentFormats', () => {
      // Arrange
      const mockWs = {
        readyState: 1, // WebSocket.OPEN
        send: vi.fn(),
      };
      const statsManager = makeMockStatsManager();
      const cameraManager = makeMockCameraManager();
      const toastManager = makeMockToastManager();
      
      const sut = new WebSocketService(statsManager, cameraManager, toastManager);
      (sut as any).ws = mockWs;

      const pngImage = new ImageData('data:image/png;base64,test', 100, 200);
      const jpegImage = new ImageData('data:image/jpeg;base64,test', 300, 400);

      // Act
      sut.sendFrameWithImageData(pngImage, ['none'], 'cpu', 'bt601');
      sut.sendFrameWithImageData(jpegImage, ['grayscale'], 'gpu', 'bt709');

      // Assert
      expect(mockWs.send).toHaveBeenCalledTimes(2);
    });
  });

  describe('Error Cases', () => {
    it('Error_InvalidImageDataThrows', () => {
      // Arrange
      const statsManager = makeMockStatsManager();
      const cameraManager = makeMockCameraManager();
      const toastManager = makeMockToastManager();
      
      const sut = new WebSocketService(statsManager, cameraManager, toastManager);

      // Act / Assert
      expect(() => {
        const invalidImage = new ImageData('', 100, 200);
        sut.sendFrameWithImageData(invalidImage, [], 'gpu', 'bt709');
      }).toThrow('Image data cannot be empty');
    });

    it('Error_InvalidDimensionsThrows', () => {
      // Arrange
      const statsManager = makeMockStatsManager();
      const cameraManager = makeMockCameraManager();
      const toastManager = makeMockToastManager();
      
      const sut = new WebSocketService(statsManager, cameraManager, toastManager);

      // Act / Assert
      expect(() => {
        const invalidImage = new ImageData('data:image/png;base64,test', -1, 200);
        sut.sendFrameWithImageData(invalidImage, [], 'gpu', 'bt709');
      }).toThrow('Image dimensions must be positive');
    });

    it('Error_WebSocketNotConnected', () => {
      // Arrange
      const mockWs = {
        readyState: 3, // WebSocket.CLOSED
        send: vi.fn(),
      };
      const statsManager = makeMockStatsManager();
      const cameraManager = makeMockCameraManager();
      const toastManager = makeMockToastManager();
      
      const sut = new WebSocketService(statsManager, cameraManager, toastManager);
      (sut as any).ws = mockWs;

      const image = makeValidImageData();

      // Act
      sut.sendFrameWithImageData(image, ['grayscale'], 'gpu', 'bt709');

      // Assert
      expect(mockWs.send).not.toHaveBeenCalled();
      expect(cameraManager.setProcessing).not.toHaveBeenCalled();
    });
  });

  describe('Edge Cases', () => {
    it('Edge_MinimumImageDimensions', () => {
      // Arrange
      const mockWs = {
        readyState: 1, // WebSocket.OPEN
        send: vi.fn(),
      };
      const statsManager = makeMockStatsManager();
      const cameraManager = makeMockCameraManager();
      const toastManager = makeMockToastManager();
      
      const sut = new WebSocketService(statsManager, cameraManager, toastManager);
      (sut as any).ws = mockWs;

      const image = new ImageData('data:image/png;base64,test', 1, 1);

      // Act
      sut.sendFrameWithImageData(image, ['none'], 'cpu', 'bt601');

      // Assert
      expect(mockWs.send).toHaveBeenCalledOnce();
    });

    it('Edge_LargeImageDimensions', () => {
      // Arrange
      const mockWs = {
        readyState: 1, // WebSocket.OPEN
        send: vi.fn(),
      };
      const statsManager = makeMockStatsManager();
      const cameraManager = makeMockCameraManager();
      const toastManager = makeMockToastManager();
      
      const sut = new WebSocketService(statsManager, cameraManager, toastManager);
      (sut as any).ws = mockWs;

      const image = new ImageData('data:image/png;base64,test', 4096, 4096);

      // Act
      sut.sendFrameWithImageData(image, ['grayscale'], 'gpu', 'bt709');

      // Assert
      expect(mockWs.send).toHaveBeenCalledOnce();
    });

    it('Edge_AllGrayscaleTypes', () => {
      // Arrange
      const mockWs = {
        readyState: 1, // WebSocket.OPEN
        send: vi.fn(),
      };
      const statsManager = makeMockStatsManager();
      const cameraManager = makeMockCameraManager();
      const toastManager = makeMockToastManager();
      
      const sut = new WebSocketService(statsManager, cameraManager, toastManager);
      (sut as any).ws = mockWs;

      const image = makeValidImageData();
      const grayscaleTypes = ['bt601', 'bt709', 'average', 'lightness', 'luminosity'];

      // Act
      grayscaleTypes.forEach(type => {
        sut.sendFrameWithImageData(image, ['grayscale'], 'gpu', type);
      });

      // Assert
      expect(mockWs.send).toHaveBeenCalledTimes(grayscaleTypes.length);
    });
  });

  describe('Table Driven Tests', () => {
    const testCases = [
      {
        name: 'Success_PngImage',
        image: () => new ImageData('data:image/png;base64,test', 100, 200),
        filters: ['grayscale'],
        accelerator: 'gpu',
        grayscaleType: 'bt709',
        shouldThrow: false
      },
      {
        name: 'Success_JpegImage',
        image: () => new ImageData('data:image/jpeg;base64,test', 300, 400),
        filters: ['none'],
        accelerator: 'cpu',
        grayscaleType: 'bt601',
        shouldThrow: false
      },
      {
        name: 'Error_EmptyImageData',
        image: () => new ImageData('', 100, 200),
        filters: ['grayscale'],
        accelerator: 'gpu',
        grayscaleType: 'bt709',
        shouldThrow: true,
        expectedError: 'Image data cannot be empty'
      },
      {
        name: 'Error_NegativeDimensions',
        image: () => new ImageData('data:image/png;base64,test', -1, 200),
        filters: ['grayscale'],
        accelerator: 'gpu',
        grayscaleType: 'bt709',
        shouldThrow: true,
        expectedError: 'Image dimensions must be positive'
      }
    ];

    testCases.forEach(({ name, image, filters, accelerator, grayscaleType, shouldThrow, expectedError }) => {
      it(name, () => {
        // Arrange
        const mockWs = {
          readyState: 1, // WebSocket.OPEN
          send: vi.fn(),
        };
        const statsManager = makeMockStatsManager();
        const cameraManager = makeMockCameraManager();
        const toastManager = makeMockToastManager();
        
        const sut = new WebSocketService(statsManager, cameraManager, toastManager);
        (sut as any).ws = mockWs;

        if (shouldThrow) {
          // Act / Assert
          expect(() => {
            const img = image();
            sut.sendFrameWithImageData(img, filters, accelerator, grayscaleType);
          }).toThrow(expectedError);
        } else {
          // Act
          const img = image();
          sut.sendFrameWithImageData(img, filters, accelerator, grayscaleType);

          // Assert
          expect(mockWs.send).toHaveBeenCalledOnce();
        }
      });
    });
  });
});

describe('WebSocketService - FilterData Integration', () => {
  const makeValidImageData = () => new ImageData('data:image/png;base64,test', 100, 200);
  const makeNoneFilter = () => new FilterData('none');
  const makeGrayscaleFilter = () => new FilterData('grayscale');

  const makeMockStatsManager = () => ({
    updateWebSocketStatus: vi.fn(),
  });

  const makeMockCameraManager = () => ({
    setProcessing: vi.fn(),
  });

  const makeMockToastManager = () => ({
    warning: vi.fn(),
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Success Cases', () => {
    it('Success_SendFrameWithFilterData', () => {
      // Arrange
      const mockWs = { readyState: 1, send: vi.fn() };
      const statsManager = makeMockStatsManager();
      const cameraManager = makeMockCameraManager();
      const toastManager = makeMockToastManager();
      
      const sut = new WebSocketService(statsManager, cameraManager, toastManager);
      (sut as any).ws = mockWs;

      const image = makeValidImageData();
      const filters = [makeNoneFilter(), makeGrayscaleFilter()];

      // Act
      sut.sendFrameWithValueObjects(image, filters, 'gpu', 'bt709');

      // Assert
      expect(mockWs.send).toHaveBeenCalledOnce();
      expect(cameraManager.setProcessing).toHaveBeenCalledWith(true);
    });

    it('Success_BackwardCompatibilityWithStringFilters', () => {
      // Arrange
      const mockWs = { readyState: 1, send: vi.fn() };
      const sut = new WebSocketService(makeMockStatsManager(), makeMockCameraManager(), makeMockToastManager());
      (sut as any).ws = mockWs;

      const image = makeValidImageData();

      // Act - usar método antiguo con strings
      sut.sendFrameWithImageData(image, ['none', 'grayscale'], 'gpu', 'bt709');

      // Assert
      expect(mockWs.send).toHaveBeenCalledOnce();
    });
  });

  const makeMockWebSocket = () => ({
    readyState: 1, // WebSocket.OPEN
    send: vi.fn(),
  });

  describe('ProcessingConfig Value Objects Integration', () => {
    describe('Success Cases', () => {
      it('Success_SendFrameWithProcessingConfig', () => {
        // Arrange
        const sut = new WebSocketService(makeMockStatsManager(), makeMockCameraManager(), makeMockToastManager());
        const mockWs = makeMockWebSocket();
        sut['ws'] = mockWs;

        const image = new ImageData('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==', 1, 1);
        const filters = [new FilterData('grayscale', { radius: 5 })];
        const accelerator = new AcceleratorConfig('cpu');
        const grayscale = new GrayscaleAlgorithm('bt601');

        // Act
        sut.sendFrameWithProcessingConfig(image, filters, accelerator, grayscale);

        // Assert
        expect(mockWs.send).toHaveBeenCalledOnce();
      });

      it('Success_CPUAccelerator', () => {
        // Arrange
        const sut = new WebSocketService(makeMockStatsManager(), makeMockCameraManager(), makeMockToastManager());
        const mockWs = makeMockWebSocket();
        sut['ws'] = mockWs;

        const image = new ImageData('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==', 1, 1);
        const filters = [new FilterData('none')];
        const accelerator = new AcceleratorConfig('cpu');
        const grayscale = new GrayscaleAlgorithm('bt601');

        // Act
        sut.sendFrameWithProcessingConfig(image, filters, accelerator, grayscale);

        // Assert
        expect(mockWs.send).toHaveBeenCalledOnce();
        expect(accelerator.isCPU()).toBe(true);
        expect(accelerator.toProtocol()).toBe(AcceleratorType.CPU);
      });

      it('Success_GPUAccelerator', () => {
        // Arrange
        const sut = new WebSocketService(makeMockStatsManager(), makeMockCameraManager(), makeMockToastManager());
        const mockWs = makeMockWebSocket();
        sut['ws'] = mockWs;

        const image = new ImageData('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==', 1, 1);
        const filters = [new FilterData('none')];
        const accelerator = new AcceleratorConfig('gpu');
        const grayscale = new GrayscaleAlgorithm('bt601');

        // Act
        sut.sendFrameWithProcessingConfig(image, filters, accelerator, grayscale);

        // Assert
        expect(mockWs.send).toHaveBeenCalledOnce();
        expect(accelerator.isGPU()).toBe(true);
        expect(accelerator.toProtocol()).toBe(AcceleratorType.CUDA);
      });

      it('Success_AllGrayscaleAlgorithms', () => {
        // Arrange
        const sut = new WebSocketService(makeMockStatsManager(), makeMockCameraManager(), makeMockToastManager());
        const mockWs = makeMockWebSocket();
        sut['ws'] = mockWs;

        const image = new ImageData('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==', 1, 1);
        const filters = [new FilterData('none')];
        const accelerator = new AcceleratorConfig('cpu');

        const algorithms = ['bt601', 'bt709', 'average', 'lightness', 'luminosity'] as const;

        algorithms.forEach(algorithm => {
          // Act
          const grayscale = new GrayscaleAlgorithm(algorithm);
          sut.sendFrameWithProcessingConfig(image, filters, accelerator, grayscale);

          // Assert
          expect(grayscale.getType()).toBe(algorithm);
          expect(grayscale.toProtocol()).toBeDefined();
        });

        expect(mockWs.send).toHaveBeenCalledTimes(algorithms.length);
      });

      it('Success_BackwardCompatibilityWithStrings', () => {
        // Arrange
        const sut = new WebSocketService(makeMockStatsManager(), makeMockCameraManager(), makeMockToastManager());
        const mockWs = makeMockWebSocket();
        sut['ws'] = mockWs;

        const image = new ImageData('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==', 1, 1);
        const filters = [new FilterData('none')];

        // Act - Test that string-based methods still work
        sut.sendFrameWithImageData(image, ['none'], 'cpu', 'bt601');

        // Assert
        expect(mockWs.send).toHaveBeenCalledOnce();
      });
    });

    describe('Error Cases', () => {
      it('Error_InvalidAcceleratorTypeThrows', () => {
        // Arrange / Act / Assert
        expect(() => {
          new AcceleratorConfig('opencl');
        }).toThrow('Invalid accelerator type');
      });

      it('Error_InvalidGrayscaleTypeThrows', () => {
        // Arrange / Act / Assert
        expect(() => {
          new GrayscaleAlgorithm('hsv');
        }).toThrow('Invalid grayscale algorithm type');
      });
    });

    describe('Edge Cases', () => {
      it('Edge_CudaNormalization', () => {
        // Arrange
        const sut = new WebSocketService(makeMockStatsManager(), makeMockCameraManager(), makeMockToastManager());
        const mockWs = makeMockWebSocket();
        sut['ws'] = mockWs;

        const image = new ImageData('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==', 1, 1);
        const filters = [new FilterData('none')];
        const accelerator = new AcceleratorConfig('cuda'); // Should normalize to 'gpu'
        const grayscale = new GrayscaleAlgorithm('bt601');

        // Act
        sut.sendFrameWithProcessingConfig(image, filters, accelerator, grayscale);

        // Assert
        expect(mockWs.send).toHaveBeenCalledOnce();
        expect(accelerator.getType()).toBe('gpu');
        expect(accelerator.isGPU()).toBe(true);
        expect(accelerator.toProtocol()).toBe(AcceleratorType.CUDA);
      });
    });
  });

  describe('Error Cases', () => {
    it('Error_InvalidFilterTypeThrows', () => {
      // Arrange
      const sut = new WebSocketService(makeMockStatsManager(), makeMockCameraManager(), makeMockToastManager());

      // Act / Assert
      expect(() => {
        new FilterData('invalid-type' as any);
      }).toThrow('Invalid filter type');
    });
  });
});
