import React from 'react';
import type { StaticImage } from '@/gen/common_pb';

type ImageSelectorModalProps = {
  isOpen: boolean;
  availableImages: StaticImage[];
  onClose: () => void;
  onSelectImage: (image: StaticImage) => void;
};

export function ImageSelectorModal({
  isOpen,
  availableImages,
  onClose,
  onSelectImage,
}: ImageSelectorModalProps) {
  return (
    <div className="react-image-modal-host" aria-hidden={!isOpen}>
      <div className={`backdrop ${isOpen ? 'show' : ''}`} onClick={onClose} />
      <div className={`modal ${isOpen ? 'show' : ''}`} data-testid="image-selector-modal">
        <div className="modal-header">
          <h2 className="modal-title">Select Image</h2>
          <button type="button" className="close-btn" onClick={onClose} data-testid="modal-close">
            {'\u00d7'}
          </button>
        </div>
        <div className="modal-content">
          {availableImages.length > 0 ? (
            <div className="image-grid">
              {availableImages.map((image) => (
                <button
                  key={image.id}
                  type="button"
                  className="image-item"
                  onClick={() => onSelectImage(image)}
                  data-testid={`image-item-${image.id}`}
                >
                  <img src={image.path} alt={image.displayName} className="image-preview" loading="lazy" />
                  <div className="image-name">{image.displayName}</div>
                  {image.isDefault ? <span className="image-badge">Default</span> : null}
                </button>
              ))}
            </div>
          ) : (
            <div className="empty-state">No images available</div>
          )}
        </div>
      </div>
    </div>
  );
}
