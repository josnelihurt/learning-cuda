import { type ReactNode, type ReactElement } from 'react';
import styles from './VideoSourceCard.module.css';

type VideoSourceCardProps = {
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

export function VideoSourceCard({
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
}: VideoSourceCardProps): ReactElement {
  const cardClassName = isSelected ? `${styles.card} ${styles.selected}` : styles.card;
  const shouldRenderProcessedImage = Boolean(imageSrc);

  return (
    <div className={styles.source}>
      <div
        className={cardClassName}
        onClick={() => onSelect(sourceId)}
        data-testid={`source-card-${sourceNumber}`}
        data-source-id={sourceId}
      >
        <div className={styles.sourceNumber}>{sourceNumber}</div>
        {sourceType === 'static' ? (
          <button
            className={styles.changeImageBtn}
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
          className={styles.closeBtn}
          onClick={(event) => {
            event.stopPropagation();
            onClose(sourceId);
          }}
          title={`Close ${sourceName}`}
          data-testid="source-close-button"
        >
          {'\u00d7'}
        </button>
        <div className={styles.content}>
          {shouldRenderProcessedImage ? (
            <img src={imageSrc} alt={sourceName} crossOrigin="anonymous" />
          ) : null}
          {children}
        </div>
      </div>
    </div>
  );
}
