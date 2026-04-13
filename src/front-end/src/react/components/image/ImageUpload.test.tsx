import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ImageUpload } from './ImageUpload';
import type { StaticImage } from '@/gen/config_service_pb';

// Mock useImageUpload hook
const mockUploadFile = vi.fn();
const mockUseImageUpload = vi.fn(() => ({
  uploading: false,
  progress: 0,
  error: null,
  uploadFile: mockUploadFile,
}));

vi.mock('../../hooks/useImageUpload', () => ({
  useImageUpload: () => mockUseImageUpload(),
}));

afterEach(() => {
  vi.clearAllMocks();
  document.body.replaceChildren();
});

describe('ImageUpload', () => {
  const setupUploadMock = (uploading = false, progress = 0, error: string | null = null) => {
    mockUseImageUpload.mockReturnValue({
      uploading,
      progress,
      error,
      uploadFile: mockUploadFile,
    });
    return mockUploadFile;
  };

  it('renders upload container with correct text', () => {
    const onImageUploaded = vi.fn();
    setupUploadMock();

    render(<ImageUpload onImageUploaded={onImageUploaded} />);

    expect(screen.getByText('Add Image')).toBeInTheDocument();
    expect(screen.getByText('Click or drag and drop to upload')).toBeInTheDocument();
    expect(screen.getByText('Only PNG files supported (max 10MB)')).toBeInTheDocument();
  });

  it('shows upload icon', () => {
    const onImageUploaded = vi.fn();
    setupUploadMock();

    render(<ImageUpload onImageUploaded={onImageUploaded} />);

    const icon = screen.getByText('+');
    expect(icon).toBeInTheDocument();
  });

  it('triggers file input click on container click', async () => {
    const onImageUploaded = vi.fn();
    setupUploadMock();

    render(<ImageUpload onImageUploaded={onImageUploaded} />);

    const container = screen.getByTestId('upload-container');
    await userEvent.click(container);

    const fileInput = screen.getByTestId('file-input') as HTMLInputElement;
    expect(fileInput).toBeInTheDocument();
  });

  it('does not trigger click when uploading', async () => {
    const onImageUploaded = vi.fn();
    setupUploadMock(true, 50);

    render(<ImageUpload onImageUploaded={onImageUploaded} />);

    const container = screen.getByTestId('upload-container');
    await userEvent.click(container);

    // CSS Modules adds a hash suffix, so check if class name contains 'uploading'
    expect(container.className).toContain('uploading');
  });

  it('shows uploading state', () => {
    const onImageUploaded = vi.fn();
    setupUploadMock(true, 50);

    render(<ImageUpload onImageUploaded={onImageUploaded} />);

    expect(screen.getByText('Uploading...')).toBeInTheDocument();
    expect(screen.getByTestId('upload-container').className).toContain('uploading');
  });

  it('shows progress bar during upload', () => {
    const onImageUploaded = vi.fn();
    setupUploadMock(true, 75);

    render(<ImageUpload onImageUploaded={onImageUploaded} />);

    const progressBar = document.querySelector('[class*="progressFill"]') as HTMLElement;
    expect(progressBar).toBeInTheDocument();
    expect(progressBar.style.width).toBe('75%');
  });

  it('shows error message when upload fails', () => {
    const onImageUploaded = vi.fn();
    setupUploadMock(false, 0, 'Upload failed');

    render(<ImageUpload onImageUploaded={onImageUploaded} />);

    expect(screen.getByText('Upload failed')).toBeInTheDocument();
    expect(screen.getByTestId('upload-error')).toBeInTheDocument();
  });

  it('shows dragging state on drag over', async () => {
    const onImageUploaded = vi.fn();
    setupUploadMock();

    render(<ImageUpload onImageUploaded={onImageUploaded} />);

    const container = screen.getByTestId('upload-container');

    fireEvent.dragOver(container);

    // CSS Modules adds a hash suffix
    expect(container.className).toContain('dragging');
  });

  it('removes dragging state on drag leave', async () => {
    const onImageUploaded = vi.fn();
    setupUploadMock();

    render(<ImageUpload onImageUploaded={onImageUploaded} />);

    const container = screen.getByTestId('upload-container');

    fireEvent.dragOver(container);
    expect(container.className).toContain('dragging');

    fireEvent.dragLeave(container);
    expect(container.className).not.toContain('dragging');
  });

  it('handles file drop', async () => {
    const onImageUploaded = vi.fn();
    const mockImage: StaticImage = {
      id: 'test-id',
      displayName: 'test.png',
      path: '/images/test.png',
      isDefault: false,
    } as StaticImage;

    const mockUploadFile = setupUploadMock();
    mockUploadFile.mockResolvedValue(mockImage);

    render(<ImageUpload onImageUploaded={onImageUploaded} />);

    const container = screen.getByTestId('upload-container');
    const file = new File(['content'], 'test.png', { type: 'image/png' });

    fireEvent.drop(container, { dataTransfer: { files: [file] } });

    await waitFor(() => {
      expect(mockUploadFile).toHaveBeenCalledWith(file);
    });

    await waitFor(() => {
      expect(onImageUploaded).toHaveBeenCalledWith(mockImage);
    });
  });

  it('handles file input change', async () => {
    const onImageUploaded = vi.fn();
    const mockImage: StaticImage = {
      id: 'test-id',
      displayName: 'test.png',
      path: '/images/test.png',
      isDefault: false,
    } as StaticImage;

    const mockUploadFile = setupUploadMock();
    mockUploadFile.mockResolvedValue(mockImage);

    render(<ImageUpload onImageUploaded={onImageUploaded} />);

    const fileInput = screen.getByTestId('file-input') as HTMLInputElement;
    const file = new File(['content'], 'test.png', { type: 'image/png' });

    await userEvent.upload(fileInput, file);

    await waitFor(() => {
      expect(mockUploadFile).toHaveBeenCalledWith(file);
    });

    await waitFor(() => {
      expect(onImageUploaded).toHaveBeenCalledWith(mockImage);
    });
  });

  it('does not handle drop when uploading', async () => {
    const onImageUploaded = vi.fn();
    const mockUploadFile = setupUploadMock(true, 50);

    render(<ImageUpload onImageUploaded={onImageUploaded} />);

    const container = screen.getByTestId('upload-container');
    const file = new File(['content'], 'test.png', { type: 'image/png' });

    fireEvent.drop(container, { dataTransfer: { files: [file] } });

    // uploadFile should not be called when uploading
    expect(mockUploadFile).not.toHaveBeenCalled();
  });
});
