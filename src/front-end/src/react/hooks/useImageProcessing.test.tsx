import { renderHook, waitFor, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { useImageProcessing } from './useImageProcessing';
import type { ActiveFilterState } from '../components/filters/FilterPanel';
import type { GrpcClients } from '../context/service-context';
import { ServiceContext } from '../context/service-context';
import { ToastContext, type ToastApi } from '../context/toast-context';
import type {
  ProcessImageRequest,
  ProcessImageResponse,
} from '@/gen/image_processor_service_pb';
import { ProcessImageResponse as ProcessImageResponseMsg } from '@/gen/image_processor_service_pb';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock image loading
const mockImageLoad = vi.fn();

describe('useImageProcessing', () => {
  // Mock gRPC client
  const mockProcessImage = vi.fn();
  const mockClient = {
    imageProcessorClient: {
      processImage: mockProcessImage,
    },
    remoteManagementClient: {} as any,
  } as GrpcClients;

  // Mock toast API
  const mockToastApi: ToastApi = {
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
  };

  // Create test wrapper
  const createWrapper = () => {
    return ({ children }: { children: React.ReactNode }) => (
      <ServiceContext.Provider value={mockClient as GrpcClients}>
        <ToastContext.Provider value={mockToastApi}>{children}</ToastContext.Provider>
      </ServiceContext.Provider>
    );
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockFetch.mockClear();
    mockImageLoad.mockClear();

    // Mock Image constructor
    global.Image = class {
      width = 100;
      height = 100;
      src = '';
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;

      constructor() {
        setTimeout(() => {
          if (this.onload) this.onload();
        }, 0);
      }
    } as any;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should initialize with default state', () => {
    const { result } = renderHook(() => useImageProcessing(), { wrapper: createWrapper() });

    expect(result.current.processing).toBe(false);
    expect(result.current.progress).toBe(0);
    expect(result.current.processedImageUrl).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it('should show error when no image path provided', async () => {
    const { result } = renderHook(() => useImageProcessing(), { wrapper: createWrapper() });

    await act(async () => {
      await result.current.processImage('', [{ id: 'filter1', parameters: { param1: 'value1' } }]);
    });

    expect(result.current.processing).toBe(false);
    expect(result.current.error).toBe('No image selected. Please upload or select an image first.');
    expect(mockToastApi.error).toHaveBeenCalledWith(
      'No Image Selected',
      'No image selected. Please upload or select an image first.'
    );
    expect(mockProcessImage).not.toHaveBeenCalled();
  });

  it('should show error when no filters provided', async () => {
    const { result } = renderHook(() => useImageProcessing(), { wrapper: createWrapper() });

    await act(async () => {
      await result.current.processImage('/path/to/image.png', []);
    });

    expect(result.current.processing).toBe(false);
    expect(result.current.error).toBe('No filters selected. Please select at least one filter.');
    expect(mockToastApi.error).toHaveBeenCalledWith(
      'No Filters Selected',
      'No filters selected. Please select at least one filter.'
    );
    expect(mockProcessImage).not.toHaveBeenCalled();
  });

  it('should show error when only "none" filter is provided', async () => {
    const { result } = renderHook(() => useImageProcessing(), { wrapper: createWrapper() });

    await act(async () => {
      await result.current.processImage('/path/to/image.png', [{ id: 'none', parameters: {} }]);
    });

    expect(result.current.processing).toBe(false);
    expect(result.current.error).toBe(
      'No filters selected. Please enable and configure at least one filter.'
    );
    expect(mockToastApi.error).toHaveBeenCalledWith(
      'No Filters Selected',
      'No filters selected. Please enable and configure at least one filter.'
    );
    expect(mockProcessImage).not.toHaveBeenCalled();
  });

  it('should process image successfully with valid inputs', async () => {
    // Setup mock fetch to return image blob
    const mockImageData = new Uint8Array([1, 2, 3, 4]);
    const mockBlob = new Blob([mockImageData], { type: 'image/png' });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      blob: async () => mockBlob,
    } as Response);

    // Setup mock gRPC response
    const mockProcessedData = new Uint8Array([5, 6, 7, 8]);
    const mockResponse = new ProcessImageResponseMsg();
    mockResponse.code = 0;
    mockResponse.message = 'Image processed successfully';
    mockResponse.imageData = mockProcessedData;
    mockResponse.width = 100;
    mockResponse.height = 100;
    mockResponse.channels = 4;
    mockProcessImage.mockResolvedValueOnce(mockResponse);

    const { result } = renderHook(() => useImageProcessing(), { wrapper: createWrapper() });

    const activeFilters: ActiveFilterState[] = [
      { id: 'filter1', parameters: { param1: 'value1' } },
    ];

    await act(async () => {
      await result.current.processImage('/path/to/image.png', activeFilters);
    });

    // Wait for processing to complete and state to update
    await waitFor(
      () => {
        expect(result.current.processing).toBe(false);
      },
      { timeout: 3000 }
    );

    // Verify successful processing
    expect(result.current.error).toBeNull();
    expect(result.current.processedImageUrl).toBeTruthy();
    expect(mockToastApi.success).toHaveBeenCalledWith(
      'Processing Complete',
      'Image processed successfully'
    );
    expect(mockProcessImage).toHaveBeenCalled();

    // Verify the request was built correctly
    const requestArg = mockProcessImage.mock.calls[0][0] as ProcessImageRequest;
    expect(requestArg.width).toBe(100);
    expect(requestArg.height).toBe(100);
    expect(requestArg.channels).toBe(4);
    expect(requestArg.genericFilters).toHaveLength(1);
    expect(requestArg.genericFilters[0].filterId).toBe('filter1');
  });

  it('should handle fetch error', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Network error'));

    const { result } = renderHook(() => useImageProcessing(), { wrapper: createWrapper() });

    const activeFilters: ActiveFilterState[] = [
      { id: 'filter1', parameters: { param1: 'value1' } },
    ];

    await result.current.processImage('/path/to/image.png', activeFilters);

    await waitFor(
      () => {
        expect(result.current.processing).toBe(false);
      },
      { timeout: 3000 }
    );

    expect(result.current.error).toContain('Network error');
    expect(mockToastApi.error).toHaveBeenCalledWith('Processing Failed', expect.any(String));
    expect(mockProcessImage).not.toHaveBeenCalled();
  });

  it('should handle gRPC processing error', async () => {
    const mockImageData = new Uint8Array([1, 2, 3, 4]);
    const mockBlob = new Blob([mockImageData], { type: 'image/png' });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      blob: async () => mockBlob,
    } as Response);

    // Mock gRPC error response
    const mockResponse = new ProcessImageResponseMsg();
    mockResponse.code = 1;
    mockResponse.message = 'CUDA processing failed';
    mockProcessImage.mockResolvedValueOnce(mockResponse);

    const { result } = renderHook(() => useImageProcessing(), { wrapper: createWrapper() });

    const activeFilters: ActiveFilterState[] = [
      { id: 'filter1', parameters: { param1: 'value1' } },
    ];

    await act(async () => {
      await result.current.processImage('/path/to/image.png', activeFilters);
    });

    await waitFor(
      () => {
        expect(result.current.processing).toBe(false);
      },
      { timeout: 3000 }
    );

    expect(result.current.error).toBe('CUDA processing failed');
    expect(mockToastApi.error).toHaveBeenCalledWith('Processing Failed', 'CUDA processing failed');
  });

  it('should reset state correctly', () => {
    const { result } = renderHook(() => useImageProcessing(), { wrapper: createWrapper() });

    // Set some state
    result.current.reset();

    expect(result.current.processedImageUrl).toBeNull();
    expect(result.current.error).toBeNull();
    expect(result.current.progress).toBe(0);
  });

  it('should convert ActiveFilterState to GenericFilterSelection correctly', async () => {
    const mockImageData = new Uint8Array([1, 2, 3, 4]);
    const mockBlob = new Blob([mockImageData], { type: 'image/png' });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      blob: async () => mockBlob,
    } as Response);

    const mockProcessedData = new Uint8Array([5, 6, 7, 8]);
    const mockResponse = new ProcessImageResponseMsg();
    mockResponse.code = 0;
    mockResponse.message = 'Success';
    mockResponse.imageData = mockProcessedData;
    mockResponse.width = 100;
    mockResponse.height = 100;
    mockResponse.channels = 4;
    mockProcessImage.mockResolvedValueOnce(mockResponse);

    const { result } = renderHook(() => useImageProcessing(), { wrapper: createWrapper() });

    const activeFilters: ActiveFilterState[] = [
      {
        id: 'grayscale',
        parameters: { type: 'luminosity', strength: '0.8' },
      },
    ];

    await act(async () => {
      await result.current.processImage('/path/to/image.png', activeFilters);
    });

    await waitFor(
      () => {
        expect(result.current.processing).toBe(false);
      },
      { timeout: 3000 }
    );

    expect(mockProcessImage).toHaveBeenCalled();
    const requestArg = mockProcessImage.mock.calls[0][0] as ProcessImageRequest;
    expect(requestArg.genericFilters).toHaveLength(1);
    expect(requestArg.genericFilters[0].filterId).toBe('grayscale');
    expect(requestArg.genericFilters[0].parameters).toHaveLength(2);

    const paramSelections = requestArg.genericFilters[0].parameters;
    expect(paramSelections[0].parameterId).toBe('type');
    expect(paramSelections[0].values).toEqual(['luminosity']);
    expect(paramSelections[1].parameterId).toBe('strength');
    expect(paramSelections[1].values).toEqual(['0.8']);
  });
});
