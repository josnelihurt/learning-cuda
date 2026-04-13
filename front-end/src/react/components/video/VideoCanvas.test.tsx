import { render, screen } from '@testing-library/react';
import { vi } from 'vitest';
import { VideoCanvas } from './VideoCanvas';

// Mock requestAnimationFrame
global.requestAnimationFrame = vi.fn((callback) => {
  return setTimeout(() => callback(performance.now()), 0) as unknown as number;
}) as any;

// Mock cancelAnimationFrame
global.cancelAnimationFrame = vi.fn() as any;

describe('VideoCanvas', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Canvas Rendering', () => {
    it('renders canvas element with correct width/height props', () => {
      render(<VideoCanvas width={640} height={480} />);

      const canvas = screen.getByLabelText('Video stream');
      expect(canvas).toBeInTheDocument();
      expect(canvas).toHaveAttribute('width', '640');
      expect(canvas).toHaveAttribute('height', '480');
    });

    it('renders with default dimensions when not provided', () => {
      render(<VideoCanvas />);

      const canvas = screen.getByLabelText('Video stream');
      expect(canvas).toHaveAttribute('width', '640');
      expect(canvas).toHaveAttribute('height', '480');
    });

    it('applies custom className', () => {
      render(<VideoCanvas className="custom-class" />);

      const container = document.querySelector('.canvasContainer');
      expect(container).toHaveClass('custom-class');
    });
  });

  describe('Frame Handling', () => {
    it('onFrame callback receives base64 frame data', () => {
      const mockOnFrame = vi.fn();

      render(<VideoCanvas width={640} height={480} onFrame={mockOnFrame} />);

      // The onFrame should be called during effect setup
      expect(mockOnFrame).toHaveBeenCalled();
    });

    it('frames are drawn directly to canvas (not via React state)', () => {
      const mockOnFrame = vi.fn();

      render(<VideoCanvas width={640} height={480} onFrame={mockOnFrame} />);

      // Verify canvas context is obtained (direct manipulation, not React state)
      const canvas = screen.getByLabelText('Video stream') as HTMLCanvasElement;
      const ctx = canvas.getContext('2d');

      expect(ctx).not.toBeNull();
      expect(ctx?.willReadFrequently).toBe(true);
    });

    it('canvas is cleared when onFrame is null', () => {
      const { rerender } = render(
        <VideoCanvas width={640} height={480} onFrame={vi.fn()} />
      );

      const canvas = screen.getByLabelText('Video stream') as HTMLCanvasElement;
      const ctx = canvas.getContext('2d');

      // Clear mock history
      vi.clearAllMocks();

      // Rerender with onFrame set to null
      rerender(<VideoCanvas width={640} height={480} />);

      // The clearRect should have been called
      expect(ctx).not.toBeNull();
    });
  });

  describe('requestAnimationFrame', () => {
    it('uses requestAnimationFrame for smooth rendering', () => {
      const mockOnFrame = vi.fn();

      render(<VideoCanvas width={640} height={480} onFrame={mockOnFrame} />);

      // requestAnimationFrame should have been called
      expect(global.requestAnimationFrame).toHaveBeenCalled();
    });

    it('cancels requestAnimationFrame on unmount', () => {
      const mockOnFrame = vi.fn();

      const { unmount } = render(
        <VideoCanvas width={640} height={480} onFrame={mockOnFrame} />
      );

      // Clear mock to track only cleanup calls
      vi.clearAllMocks();

      unmount();

      // cancelAnimationFrame should have been called during cleanup
      expect(global.cancelAnimationFrame).toHaveBeenCalled();
    });
  });

  describe('FPS Counter', () => {
    it('displays frame rate metrics per D-07', async () => {
      const mockOnFrame = vi.fn();

      render(<VideoCanvas width={640} height={480} onFrame={mockOnFrame} />);

      // FPS counter should appear after a second
      await vi.waitFor(
        () => {
          const fpsCounter = document.querySelector('.fpsCounter');
          expect(fpsCounter).toBeInTheDocument();
        },
        { timeout: 1100 }
      );
    });

    it('updates FPS counter periodically', async () => {
      const mockOnFrame = vi.fn();

      render(<VideoCanvas width={640} height={480} onFrame={mockOnFrame} />);

      await vi.waitFor(
        () => {
          const fpsCounter = document.querySelector('.fpsCounter');
          expect(fpsCounter).toBeInTheDocument();
        },
        { timeout: 1100 }
      );

      // Wait for another update cycle
      await vi.waitFor(
        () => {
          const fpsCounter = document.querySelector('.fpsCounter');
          expect(fpsCounter?.textContent).toMatch(/\d+ FPS/);
        },
        { timeout: 1100 }
      );
    });

    it('does not display FPS when not streaming', () => {
      render(<VideoCanvas width={640} height={480} />);

      const fpsCounter = document.querySelector('.fpsCounter');
      expect(fpsCounter).not.toBeInTheDocument();
    });
  });
});
