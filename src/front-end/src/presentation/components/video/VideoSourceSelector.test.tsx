import { render, screen, fireEvent } from '@testing-library/react';
import { vi } from 'vitest';
import { VideoSourceSelector } from './VideoSourceSelector';
import type { StaticImage } from '@/gen/common_pb';

describe('VideoSourceSelector', () => {
  const mockVideos: StaticImage[] = [
    {
      id: 'video1',
      name: 'Test Video 1',
      width: 640,
      height: 480,
      format: 'mp4',
      createdAt: { seconds: 1, nanos: 0 },
      updatedAt: { seconds: 1, nanos: 0 },
    },
    {
      id: 'video2',
      name: 'Test Video 2',
      width: 1280,
      height: 720,
      format: 'mp4',
      createdAt: { seconds: 2, nanos: 0 },
      updatedAt: { seconds: 2, nanos: 0 },
    },
  ];

  const mockOnSourceChange = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Tab Rendering', () => {
    it('renders camera and file tabs per D-08', () => {
      render(
        <VideoSourceSelector
          availableVideos={mockVideos}
          selectedVideoId={undefined}
          onSourceChange={mockOnSourceChange}
        />
      );

      expect(screen.getByRole('button', { name: /camera/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /file/i })).toBeInTheDocument();
    });

    it('camera tab is active by default', () => {
      render(
        <VideoSourceSelector
          availableVideos={mockVideos}
          selectedVideoId={undefined}
          onSourceChange={mockOnSourceChange}
        />
      );

      const cameraTab = screen.getByRole('button', { name: /camera/i });
      expect(cameraTab.className).toContain('tabActive');
    });
  });

  describe('Tab Switching', () => {
    it('clicking camera tab sets sourceType to camera', () => {
      render(
        <VideoSourceSelector
          availableVideos={mockVideos}
          selectedVideoId={undefined}
          onSourceChange={mockOnSourceChange}
        />
      );

      const fileTab = screen.getByRole('button', { name: /file/i });
      fireEvent.click(fileTab);

      const cameraTab = screen.getByRole('button', { name: /camera/i });
      fireEvent.click(cameraTab);

      expect(mockOnSourceChange).toHaveBeenCalledWith({ type: 'camera' });
    });

    it('clicking file tab sets sourceType to file', () => {
      render(
        <VideoSourceSelector
          availableVideos={mockVideos}
          selectedVideoId={undefined}
          onSourceChange={mockOnSourceChange}
        />
      );

      const fileTab = screen.getByRole('button', { name: /file/i });
      fireEvent.click(fileTab);

      // Should not call onSourceChange yet (file selection happens via FileList)
      expect(mockOnSourceChange).not.toHaveBeenCalled();
    });

    it('active tab uses accent color per UI-SPEC', () => {
      render(
        <VideoSourceSelector
          availableVideos={mockVideos}
          selectedVideoId={undefined}
          onSourceChange={mockOnSourceChange}
        />
      );

      const cameraTab = screen.getByRole('button', { name: /camera/i });
      expect(cameraTab.className).toContain('tabActive');
    });
  });

  describe('FileList Integration', () => {
    it('file tab shows FileList component per D-09', () => {
      render(
        <VideoSourceSelector
          availableVideos={mockVideos}
          selectedVideoId={undefined}
          onSourceChange={mockOnSourceChange}
        />
      );

      // Initially, FileList is not visible (camera tab is active)
      expect(screen.queryByTestId('file-list')).not.toBeInTheDocument();

      // Click file tab
      const fileTab = screen.getByRole('button', { name: /file/i });
      fireEvent.click(fileTab);

      // Now FileList should be visible
      const fileList = screen.queryByTestId('file-list');
      expect(fileList).toBeInTheDocument();
    });

    it('FileList onImageSelect updates selectedVideoId per D-10', () => {
      render(
        <VideoSourceSelector
          availableVideos={mockVideos}
          selectedVideoId={undefined}
          onSourceChange={mockOnSourceChange}
        />
      );

      // Click file tab
      const fileTab = screen.getByRole('button', { name: /file/i });
      fireEvent.click(fileTab);

      // Click on a video in the FileList
      const videoItem = screen.getByTestId('image-item-video1');
      fireEvent.click(videoItem);

      expect(mockOnSourceChange).toHaveBeenCalledWith({
        type: 'file',
        id: 'video1',
      });
    });
  });

  describe('Custom Styling', () => {
    it('applies custom className', () => {
      const { container } = render(
        <VideoSourceSelector
          availableVideos={mockVideos}
          selectedVideoId={undefined}
          onSourceChange={mockOnSourceChange}
          className="custom-class"
        />
      );

      const selector = container.querySelector('.custom-class');
      expect(selector).toBeInTheDocument();
    });
  });
});
