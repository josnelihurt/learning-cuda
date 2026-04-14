import React from 'react';
import type { InputSource } from '@/gen/config_service_pb';
import { useState } from 'react';
import { ImageUpload } from '../image/ImageUpload';
import { ReactVideoUpload } from './ReactVideoUpload';
import { ReactVideoSelector } from './ReactVideoSelector';

type ReactSourceDrawerProps = {
  isOpen: boolean;
  availableSources: InputSource[];
  onClose: () => void;
  onSelectSource: (source: InputSource) => void;
  onSourcesChanged: () => void;
};

export function ReactSourceDrawer({
  isOpen,
  availableSources,
  onClose,
  onSelectSource,
  onSourcesChanged,
}: ReactSourceDrawerProps) {
  const [activeTab, setActiveTab] = useState<'images' | 'videos'>('images');
  const [videoReloadKey, setVideoReloadKey] = useState(0);
  const filteredSources =
    activeTab === 'images'
      ? availableSources.filter((source) => source.type === 'static' || source.type === 'camera')
      : [];

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
              className={`tab ${activeTab === 'videos' ? 'active' : ''}`}
              onClick={() => setActiveTab('videos')}
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
                    <div className="source-icon">{source.type === 'camera' ? '\u25cf' : '\u25a3'}</div>
                    <div className="source-info">
                      <div className="source-name">{source.displayName}</div>
                      <div className="source-type">{source.type}</div>
                    </div>
                    {source.isDefault ? <span className="source-badge">Default</span> : null}
                  </div>
                ))}
              </div>
            </>
          ) : (
            <>
              <div className="upload-section">
                <div className="section-title">Upload Video</div>
                <ReactVideoUpload
                  onVideoUploaded={() => {
                    onSourcesChanged();
                    setVideoReloadKey((current) => current + 1);
                  }}
                />
              </div>
              <div className="section-title">Select Video</div>
              <ReactVideoSelector
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
