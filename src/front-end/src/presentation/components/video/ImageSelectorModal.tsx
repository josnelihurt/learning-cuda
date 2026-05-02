import React, { useCallback, useEffect, useRef, useState } from 'react';
import type { CapturedImageInfo } from '@/gen/image_processor_service_pb';
import { CapturedImageFormat } from '@/gen/image_processor_service_pb';
import type { StaticImage } from '@/gen/common_pb';
import { controlChannelService } from '@/infrastructure/transport/control-channel-service';
import './video-grid.css';

type Tab = 'library' | 'captured';

type ImageSelectorModalProps = {
  isOpen: boolean;
  availableImages: StaticImage[];
  onClose: () => void;
  onSelectImage: (image: StaticImage) => void;
};

type CapturedThumbProps = {
  id: string;
};

function CapturedThumb({ id }: CapturedThumbProps): React.ReactElement {
  const [src, setSrc] = useState<string | null>(null);
  const blobUrlRef = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    controlChannelService.getCapturedImage(id).then((resp) => {
      if (cancelled) return;
      if (resp.found && resp.imageData.length > 0) {
        const blob = new Blob([resp.imageData], { type: 'image/png' });
        const url = URL.createObjectURL(blob);
        blobUrlRef.current = url;
        setSrc(url);
      }
    }).catch(() => {
      // thumbnail fetch failed silently — skeleton stays
    });
    return () => {
      cancelled = true;
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
    };
  }, [id]);

  if (src) {
    return <img src={src} alt={id} className="image-preview" loading="lazy" />;
  }
  return <div className="image-thumb-skeleton" />;
}

export function ImageSelectorModal({
  isOpen,
  availableImages,
  onClose,
  onSelectImage,
}: ImageSelectorModalProps): React.ReactElement {
  const [activeTab, setActiveTab] = useState<Tab>('library');
  const [capturedImages, setCapturedImages] = useState<CapturedImageInfo[]>([]);
  const [capturedPage, setCapturedPage] = useState(0);
  const [capturedHasMore, setCapturedHasMore] = useState(true);
  const [capturedLoading, setCapturedLoading] = useState(false);
  const sentinelRef = useRef<HTMLDivElement>(null);

  const loadNextCapturedPage = useCallback(async (page: number): Promise<void> => {
    if (capturedLoading) return;
    setCapturedLoading(true);
    try {
      const resp = await controlChannelService.listCapturedImages(page);
      setCapturedImages((prev) => [...prev, ...resp.images]);
      setCapturedPage(page + 1);
      setCapturedHasMore(resp.hasMore);
    } catch {
      // list failed — stop pagination
      setCapturedHasMore(false);
    } finally {
      setCapturedLoading(false);
    }
  }, [capturedLoading]);

  const handleDownload = useCallback(async (id: string, filename: string): Promise<void> => {
    try {
      const resp = await controlChannelService.getCapturedImage(id, 0, 0, CapturedImageFormat.BMP);
      if (!resp.found) return;
      const blob = new Blob([resp.imageData], { type: 'image/bmp' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      // download failed silently
    }
  }, []);

  const handleDelete = useCallback(async (id: string): Promise<void> => {
    try {
      const resp = await controlChannelService.deleteCapturedImage(id);
      if (resp.deleted) {
        setCapturedImages((prev) => prev.filter((img) => img.id !== id));
      }
    } catch {
      // delete failed silently
    }
  }, []);

  // Load first page when tab becomes active (and we have no data yet)
  useEffect(() => {
    if (activeTab === 'captured' && isOpen && capturedImages.length === 0 && !capturedLoading) {
      void loadNextCapturedPage(0);
    }
  }, [activeTab, isOpen, capturedImages.length, capturedLoading, loadNextCapturedPage]);

  // Reset captured state when modal closes
  useEffect(() => {
    if (!isOpen) {
      setCapturedImages([]);
      setCapturedPage(0);
      setCapturedHasMore(true);
      setCapturedLoading(false);
    }
  }, [isOpen]);

  // IntersectionObserver for infinite scroll sentinel
  useEffect(() => {
    if (activeTab !== 'captured' || !sentinelRef.current) return;
    const sentinel = sentinelRef.current;
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && capturedHasMore && !capturedLoading) {
          void loadNextCapturedPage(capturedPage);
        }
      },
      { threshold: 0.1 },
    );
    obs.observe(sentinel);
    return () => obs.disconnect();
  }, [activeTab, capturedHasMore, capturedLoading, capturedPage, loadNextCapturedPage]);

  return (
    <div className="react-image-modal-host" aria-hidden={!isOpen}>
      <div className={`backdrop ${isOpen ? 'show' : ''}`} onClick={onClose} />
      <div className={`modal ${isOpen ? 'show' : ''}`} data-testid="image-selector-modal">
        <div className="modal-header">
          <h2 className="modal-title">Select Image</h2>
          <button type="button" className="close-btn" onClick={onClose} data-testid="modal-close">
            {'×'}
          </button>
        </div>
        <div className="modal-tabs">
          <button
            type="button"
            className={`modal-tab ${activeTab === 'library' ? 'active' : ''}`}
            onClick={() => { setActiveTab('library'); }}
            data-testid="tab-library"
          >
            Library
          </button>
          <button
            type="button"
            className={`modal-tab ${activeTab === 'captured' ? 'active' : ''}`}
            onClick={() => { setActiveTab('captured'); }}
            data-testid="tab-captured"
          >
            Captured
          </button>
        </div>
        <div className="modal-content">
          {activeTab === 'library' && (
            availableImages.length > 0 ? (
              <div className="image-grid">
                {availableImages.map((image) => (
                  <button
                    key={image.id}
                    type="button"
                    className="image-item"
                    onClick={() => { onSelectImage(image); }}
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
            )
          )}
          {activeTab === 'captured' && (
            <>
              {capturedImages.length > 0 ? (
                <div className="image-grid">
                  {capturedImages.map((img) => (
                    <div key={img.id} className="image-item image-item--captured" data-testid={`captured-item-${img.id}`}>
                      <div className="image-item-thumb">
                        <CapturedThumb id={img.id} />
                        <div className="image-item-actions">
                          <button
                            type="button"
                            className="image-action-btn"
                            title="Download BMP"
                            onClick={() => { void handleDownload(img.id, img.filename); }}
                            data-testid={`download-${img.id}`}
                          >
                            ⬇
                          </button>
                          <button
                            type="button"
                            className="image-action-btn image-action-btn--danger"
                            title="Delete"
                            onClick={() => { void handleDelete(img.id); }}
                            data-testid={`delete-${img.id}`}
                          >
                            🗑
                          </button>
                        </div>
                      </div>
                      <div className="image-name" title={img.filename}>{img.id}</div>
                    </div>
                  ))}
                </div>
              ) : !capturedLoading ? (
                <div className="empty-state">No captured images yet — click Capture to save a frame</div>
              ) : null}
              {capturedLoading && (
                <div className="captured-loading">Loading…</div>
              )}
              <div ref={sentinelRef} className="scroll-sentinel" aria-hidden="true" />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
