import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { SourceDrawer } from './SourceDrawer';

vi.mock('../image/ImageUpload', () => ({
  ImageUpload: () => <div data-testid="mock-image-upload" />,
}));

vi.mock('./VideoUpload', () => ({
  VideoUpload: () => <div data-testid="mock-video-upload" />,
}));

vi.mock('./VideoSelector', () => ({
  VideoSelector: () => <div data-testid="mock-video-selector" />,
}));

describe('SourceDrawer', () => {
  it('shows static and camera sources in images tab', () => {
    const onSelectSource = vi.fn();
    render(
      <SourceDrawer
        isOpen
        availableSources={[
          { id: 's1', displayName: 'Lena', type: 'static', imagePath: '', isDefault: true },
          { id: 's2', displayName: 'Cam', type: 'camera', imagePath: '', isDefault: false },
          { id: 's3', displayName: 'Video', type: 'video', imagePath: '', isDefault: false },
        ]}
        onClose={vi.fn()}
        onSelectSource={onSelectSource}
        onSourcesChanged={vi.fn()}
      />
    );

    expect(screen.getByTestId('source-item-s1')).toBeInTheDocument();
    expect(screen.getByTestId('source-item-s2')).toBeInTheDocument();
    expect(screen.queryByTestId('source-item-s3')).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId('source-item-s1'));
    expect(onSelectSource).toHaveBeenCalled();
  });

  it('switches to videos tab and filters sources', () => {
    render(
      <SourceDrawer
        isOpen
        availableSources={[
          { id: 's1', displayName: 'Lena', type: 'static', imagePath: '', isDefault: true },
          { id: 's3', displayName: 'Video', type: 'video', imagePath: '', isDefault: false },
        ]}
        onClose={vi.fn()}
        onSelectSource={vi.fn()}
        onSourcesChanged={vi.fn()}
      />
    );

    fireEvent.click(screen.getByTestId('tab-videos'));
    expect(screen.getByText('Select Video')).toBeInTheDocument();
    expect(screen.queryByTestId('source-item-s1')).not.toBeInTheDocument();
  });
});
