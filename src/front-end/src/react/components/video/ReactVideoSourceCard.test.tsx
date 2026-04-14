import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ReactVideoSourceCard } from './ReactVideoSourceCard';

describe('ReactVideoSourceCard', () => {
  it('calls select and close handlers', () => {
    const onSelect = vi.fn();
    const onClose = vi.fn();
    const onChangeImage = vi.fn();

    render(
      <ReactVideoSourceCard
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

    const { rerender } = render(<ReactVideoSourceCard {...baseProps} sourceType="static" />);
    fireEvent.click(screen.getByTestId('change-image-button'));
    expect(onChangeImage).toHaveBeenCalledWith('source-1', 1);

    rerender(<ReactVideoSourceCard {...baseProps} sourceType="camera" />);
    expect(screen.queryByTestId('change-image-button')).not.toBeInTheDocument();
  });
});
