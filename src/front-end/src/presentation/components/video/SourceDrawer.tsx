import React from 'react';
import type { InputSource } from '@/gen/config_service_pb';
import './video-grid.css';
import { useEffect, useMemo, useState } from 'react';
import {
  effectiveAutoselectSourceId,
  gridSourceDisplayName,
  isWebcamUsable,
} from '@/presentation/utils/input-source-defaults';
import { ImageUpload } from '@/presentation/components/image/ImageUpload';
import { VideoUpload } from './VideoUpload';
import { VideoSelector } from './VideoSelector';
import { ToastContext } from '@/presentation/context/toast-context';
import { useContext } from 'react';

type SourceDrawerProps = {
  isOpen: boolean;
  availableSources: InputSource[];
  onClose: () => void;
  onSelectSource: (source: InputSource) => void;
  onSourcesChanged: () => void;
};

export function SourceDrawer({
  isOpen,
  availableSources,
  onClose,
  onSelectSource,
  onSourcesChanged,
}: SourceDrawerProps) {
  const toast = useContext(ToastContext);
  const [activeTab, setActiveTab] = useState<'images' | 'videos'>('images');
  const [videoReloadKey, setVideoReloadKey] = useState(0);
  const [autoselectSourceId, setAutoselectSourceId] = useState<string | null>(null);
  const filteredSources = useMemo(
    () =>
      activeTab === 'images'
        ? availableSources.filter(
            (source) =>
              source.type === 'static' ||
              source.type === 'camera' ||
              source.type === 'remote_camera'
          )
        : [],
    [activeTab, availableSources]
  );

  useEffect(() => {
    if (!isOpen || activeTab !== 'images') {
      return;
    }
    let cancelled = false;
    void (async () => {
      const usable = await isWebcamUsable();
      if (cancelled) return;
      const id = effectiveAutoselectSourceId(filteredSources, usable);
      setAutoselectSourceId(id ?? null);
    })();
    return () => {
      cancelled = true;
    };
  }, [isOpen, activeTab, filteredSources]);

  return (
    <div className="react-source-drawer-host" aria-hidden={!isOpen}>
      <div className={`backdrop ${isOpen ? 'show' : ''}`} onClick={onClose} />
      <div className={`drawer ${isOpen ? 'show' : ''}`} data-testid="source-drawer">
        <div className="drawer-header">
          <h2 className="drawer-title">Select Input Source</h2>
          <button type="button" className="close-btn" onClick={onClose} data-testid="drawer-close">
            {'\u00d7'}
          </button>
        </div>
        <div className="drawer-content">
          <div className="tabs">
            <button
              type="button"
              className={`tab ${activeTab === 'images' ? 'active' : ''}`}
              onClick={() => setActiveTab('images')}
              data-testid="tab-images"
            >
              Images
            </button>
            <button
              type="button"
              className="tab tab-disabled"
              onClick={(event) => {
                event.preventDefault();
                toast?.warning(
                  'Not available',
                  'Not available in this version. Like and subscribe!'
                );
              }}
              data-testid="tab-videos"
            >
              Videos
            </button>
          </div>
          {activeTab === 'images' ? (
            <>
              <div className="upload-section">
                <div className="section-title">Upload Image</div>
                <ImageUpload
                  onImageUploaded={() => {
                    onSourcesChanged();
                  }}
                />
              </div>
              <div className="section-title">Select Source</div>
              <div className="source-list">
                {filteredSources.map((source) => (
                  <div
                    key={source.id}
                    className="source-item"
                    data-testid={`source-item-${source.id}`}
                    onClick={() => onSelectSource(source)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        onSelectSource(source);
                      }
                    }}
                  >
                    <div className="source-icon">
                      {source.type === 'camera' || source.type === 'remote_camera' ? '\u25cf' : '\u25a3'}
                    </div>
                    <div className="source-info">
                      <div className="source-name">{gridSourceDisplayName(source)}</div>
                      <div className="source-type">
                        {source.type === 'remote_camera'
                          ? 'Remote Camera'
                          : source.type === 'camera'
                            ? 'Camera'
                            : source.type}
                      </div>
                    </div>
                    {autoselectSourceId === source.id ? (
                      <span className="source-badge">Default</span>
                    ) : null}
                  </div>
                ))}
              </div>
            </>
          ) : (
            <>
              <div className="upload-section">
                <div className="section-title">Upload Video</div>
                <VideoUpload
                  onVideoUploaded={() => {
                    onSourcesChanged();
                    setVideoReloadKey((current) => current + 1);
                  }}
                />
              </div>
              <div className="section-title">Select Video</div>
              <VideoSelector
                reloadKey={videoReloadKey}
                onVideoSelected={(source) => {
                  onSelectSource(source);
                }}
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
