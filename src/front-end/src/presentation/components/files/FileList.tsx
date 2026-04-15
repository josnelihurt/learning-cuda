import type { StaticImage } from '@/gen/common_pb';
import styles from './FileList.module.css';

interface FileListProps {
  images: StaticImage[];
  selectedImageId?: string;
  onImageSelect: (image: StaticImage) => void;
  layout?: 'list' | 'grid';
  loading?: boolean;
}

export function FileList({
  images,
  selectedImageId,
  onImageSelect,
  layout = 'grid',
  loading = false,
}: FileListProps) {
  if (loading) {
    return (
      <div className={styles.container} data-testid="file-list-loading">
        <div className={styles.loading}>Loading images...</div>
      </div>
    );
  }

  if (images.length === 0) {
    return (
      <div className={styles.container} data-testid="file-list-empty">
        <div className={styles.empty}>No images available</div>
      </div>
    );
  }

  const containerClass = layout === 'grid' ? styles.grid : styles.list;

  return (
    <div
      className={`${styles.container} ${containerClass}`}
      data-testid="file-list"
      data-layout={layout}
    >
      {images.map((image) => (
        <div
          key={image.id}
          className={`${styles.imageItem} ${selectedImageId === image.id ? styles.selected : ''}`}
          onClick={() => onImageSelect(image)}
          data-testid={`image-item-${image.id}`}
        >
          <img
            src={image.path}
            alt={image.displayName}
            className={styles.thumbnail}
            loading="lazy"
          />
          <div className={styles.imageName}>{image.displayName}</div>
          {image.isDefault && <span className={styles.badge}>Default</span>}
        </div>
      ))}
    </div>
  );
}
