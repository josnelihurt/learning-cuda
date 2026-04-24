import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { SourceDrawer } from './SourceDrawer';
import { ToastProvider } from '@/presentation/context/toast-context';

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
      <ToastProvider>
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
      </ToastProvider>
    );

    expect(screen.getByTestId('source-item-s1')).toBeInTheDocument();
    expect(screen.getByTestId('source-item-s2')).toBeInTheDocument();
    expect(screen.queryByTestId('source-item-s3')).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId('source-item-s1'));
    expect(onSelectSource).toHaveBeenCalled();
  });

  it('shows disabled videos tab with toast on click', () => {
    render(
      <ToastProvider>
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
      </ToastProvider>
    );

    const videosTab = screen.getByTestId('tab-videos');
    fireEvent.click(videosTab);
    expect(screen.getByText('Not available')).toBeInTheDocument();
    expect(screen.getByText('Not available in this version. Like and subscribe!')).toBeInTheDocument();
    expect(screen.getByText('Select Source')).toBeInTheDocument();
  });
});
