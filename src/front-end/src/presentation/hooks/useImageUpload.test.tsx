import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useImageUpload } from './useImageUpload';
import type { StaticImage } from '@/gen/config_service_pb';

// Mock file-service at module level
vi.mock('@/infrastructure/data/file-service', () => ({
  fileService: {
    uploadImage: vi.fn(),
  },
}));

// Mock useToast at module level
const mockToastError = vi.fn();
vi.mock('./useToast', () => ({
  useToast: () => ({
    error: mockToastError,
  }),
}));

afterEach(() => {
  vi.clearAllMocks();
  document.body.replaceChildren();
});

describe('useImageUpload', () => {
  it('should initialize with default state', () => {
    const { result } = renderHook(() => useImageUpload());

    expect(result.current.uploading).toBe(false);
    expect(result.current.progress).toBe(0);
    expect(result.current.error).toBe(null);
  });

  it('should reject non-PNG files', async () => {
    const { result } = renderHook(() => useImageUpload());
    const file = new File(['content'], 'test.jpg', { type: 'image/jpeg' });

    let uploadResult;
    await act(async () => {
      uploadResult = await result.current.uploadFile(file);
    });

    expect(result.current.uploading).toBe(false);
    expect(result.current.error).toBe('Only PNG files are supported');
    expect(mockToastError).toHaveBeenCalledWith('Invalid File', 'Only PNG files are supported');
    expect(uploadResult).toBeUndefined();
  });

  it('should reject files larger than 10MB', async () => {
    const { result } = renderHook(() => useImageUpload());
    const largeFile = new File(['x'.repeat(11 * 1024 * 1024)], 'test.png', {
      type: 'image/png',
    });

    let uploadResult;
    await act(async () => {
      uploadResult = await result.current.uploadFile(largeFile);
    });

    expect(result.current.uploading).toBe(false);
    expect(result.current.error).toBe('File size exceeds 10MB limit');
    expect(mockToastError).toHaveBeenCalledWith('File Too Large', 'File size exceeds 10MB limit');
    expect(uploadResult).toBeUndefined();
  });

  it('should upload valid PNG file successfully', async () => {
    const mockImage: StaticImage = {
      id: 'test-id',
      displayName: 'test.png',
      path: '/images/test.png',
      isDefault: false,
    } as StaticImage;

    const { fileService } = await import('@/infrastructure/data/file-service');
    vi.mocked(fileService.uploadImage).mockResolvedValue(mockImage);

    const { result } = renderHook(() => useImageUpload());
    const file = new File(['content'], 'test.png', { type: 'image/png' });

    let uploadResult;
    await act(async () => {
      uploadResult = await result.current.uploadFile(file);
    });

    expect(fileService.uploadImage).toHaveBeenCalledWith(file);

    // Wait for progress simulation and state reset
    await waitFor(
      () => {
        expect(result.current.uploading).toBe(false);
        expect(result.current.progress).toBe(0);
      },
      { timeout: 1000 }
    );

    expect(uploadResult).toEqual(mockImage);
    expect(result.current.error).toBe(null);
  });

  it('should handle upload errors', async () => {
    const uploadError = new Error('Network error');
    const { fileService } = await import('@/infrastructure/data/file-service');
    vi.mocked(fileService.uploadImage).mockRejectedValue(uploadError);

    const { result } = renderHook(() => useImageUpload());
    const file = new File(['content'], 'test.png', { type: 'image/png' });

    let uploadResult;
    await act(async () => {
      uploadResult = await result.current.uploadFile(file);
    });

    await waitFor(
      () => {
        expect(result.current.uploading).toBe(false);
      },
      { timeout: 1000 }
    );

    expect(result.current.error).toBe('Network error');
    expect(mockToastError).toHaveBeenCalledWith('Upload Failed', 'Network error');
    expect(uploadResult).toBeUndefined();
  });

  it('should update progress during upload', async () => {
    const mockImage: StaticImage = {
      id: 'test-id',
      displayName: 'test.png',
      path: '/images/test.png',
      isDefault: false,
    } as StaticImage;

    const { fileService } = await import('@/infrastructure/data/file-service');
    vi.mocked(fileService.uploadImage).mockImplementation(
      () =>
        new Promise((resolve) => {
          setTimeout(() => resolve(mockImage), 500);
        })
    );

    const { result } = renderHook(() => useImageUpload());
    const file = new File(['content'], 'test.png', { type: 'image/png' });

    act(() => {
      result.current.uploadFile(file);
    });

    // Progress should be > 0 during upload
    await waitFor(
      () => {
        expect(result.current.progress).toBeGreaterThan(0);
      },
      { timeout: 500 }
    );

    // Should reach 100% after upload completes
    await waitFor(
      () => {
        expect(result.current.progress).toBe(100);
      },
      { timeout: 1000 }
    );
  });
});
