import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { VideoSourceCard } from './VideoSourceCard';

describe('VideoSourceCard', () => {
  it('calls select and close handlers', () => {
    const onSelect = vi.fn();
    const onClose = vi.fn();
    const onChangeImage = vi.fn();

    render(
      <VideoSourceCard
        sourceId="source-1"
        sourceNumber={1}
        sourceName="Lena"
        sourceType="static"
        imageSrc="/image.png"
        isSelected={false}
        onSelect={onSelect}
        onClose={onClose}
        onChangeImage={onChangeImage}
      />
    );

    fireEvent.click(screen.getByTestId('source-card-1'));
    expect(onSelect).toHaveBeenCalledWith('source-1');

    fireEvent.click(screen.getByTestId('source-close-button'));
    expect(onClose).toHaveBeenCalledWith('source-1');
  });

  it('triggers change image for static source only', () => {
    const onChangeImage = vi.fn();
    const baseProps = {
      sourceId: 'source-1',
      sourceNumber: 1,
      sourceName: 'Lena',
      imageSrc: '/image.png',
      isSelected: false,
      onSelect: vi.fn(),
      onClose: vi.fn(),
      onChangeImage,
    };

    const { rerender } = render(<VideoSourceCard {...baseProps} sourceType="static" />);
    fireEvent.click(screen.getByTestId('change-image-button'));
    expect(onChangeImage).toHaveBeenCalledWith('source-1', 1);

    rerender(<VideoSourceCard {...baseProps} sourceType="camera" />);
    expect(screen.queryByTestId('change-image-button')).not.toBeInTheDocument();
  });

  it('shows fps badge for video source', () => {
    render(
      <VideoSourceCard
        sourceId="video-1"
        sourceNumber={2}
        sourceName="Sample video"
        sourceType="video"
        imageSrc="/video-preview.png"
        fps={23.456}
        displayWidth={1280}
        displayHeight={720}
        isSelected={false}
        onSelect={vi.fn()}
        onClose={vi.fn()}
        onChangeImage={vi.fn()}
      />
    );

    expect(screen.getByTestId('source-fps-2')).toHaveTextContent('23.5 FPS | 1280x720');
  });

  it('shows fps badge for camera source', () => {
    render(
      <VideoSourceCard
        sourceId="camera-1"
        sourceNumber={3}
        sourceName="Camera"
        sourceType="camera"
        imageSrc=""
        fps={30}
        displayWidth={640}
        displayHeight={480}
        isSelected={false}
        onSelect={vi.fn()}
        onClose={vi.fn()}
        onChangeImage={vi.fn()}
      />
    );

    expect(screen.getByTestId('source-fps-3')).toHaveTextContent('30.0 FPS | 640x480');
  });

  it('shows resolution fallback when unavailable', () => {
    render(
      <VideoSourceCard
        sourceId="video-2"
        sourceNumber={5}
        sourceName="Sample video fallback"
        sourceType="video"
        imageSrc="/video-preview.png"
        fps={12}
        isSelected={false}
        onSelect={vi.fn()}
        onClose={vi.fn()}
        onChangeImage={vi.fn()}
      />
    );

    expect(screen.getByTestId('source-fps-5')).toHaveTextContent('12.0 FPS | --');
  });

  it('does not show fps badge for static source', () => {
    render(
      <VideoSourceCard
        sourceId="static-1"
        sourceNumber={4}
        sourceName="Static"
        sourceType="static"
        imageSrc="/img.png"
        fps={30}
        isSelected={false}
        onSelect={vi.fn()}
        onClose={vi.fn()}
        onChangeImage={vi.fn()}
      />
    );

    expect(screen.queryByTestId('source-fps-4')).not.toBeInTheDocument();
  });
});
