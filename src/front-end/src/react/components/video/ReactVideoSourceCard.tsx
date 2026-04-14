import type { ReactNode } from 'react';

type ReactVideoSourceCardProps = {
  sourceId: string;
  sourceNumber: number;
  sourceName: string;
  sourceType: string;
  imageSrc: string;
  isSelected: boolean;
  onSelect: (sourceId: string) => void;
  onClose: (sourceId: string) => void;
  onChangeImage: (sourceId: string, sourceNumber: number) => void;
  children?: ReactNode;
};

export function ReactVideoSourceCard({
  sourceId,
  sourceNumber,
  sourceName,
  sourceType,
  imageSrc,
  isSelected,
  onSelect,
  onClose,
  onChangeImage,
  children,
}: ReactVideoSourceCardProps) {
  const cardClassName = isSelected ? 'card selected' : 'card';

  return (
    <div className="react-video-source">
      <div
        className={cardClassName}
        onClick={() => onSelect(sourceId)}
        data-testid={`source-card-${sourceNumber}`}
        data-source-id={sourceId}
      >
        <div className="source-number">{sourceNumber}</div>
        {sourceType === 'static' ? (
          <button
            className="change-image-btn"
            onClick={(event) => {
              event.stopPropagation();
              onChangeImage(sourceId, sourceNumber);
            }}
            title="Change image"
            data-testid="change-image-button"
          >
            {'\u27f3'}
          </button>
        ) : null}
        <button
          className="close-btn"
          onClick={(event) => {
            event.stopPropagation();
            onClose(sourceId);
          }}
          title={`Close ${sourceName}`}
          data-testid="source-close-button"
        >
          {'\u00d7'}
        </button>
        <div className="content">
          {imageSrc ? <img src={imageSrc} alt={sourceName} crossOrigin="anonymous" /> : children}
        </div>
      </div>
    </div>
  );
}
