import { CameraPreview } from './CameraPreview';
import { VideoSourceCard } from './VideoSourceCard';
import './video-grid.css';

export type GridSource = {
  id: string;
  number: number;
  name: string;
  type: string;
  imageSrc: string;
};

type CameraFramePayload = {
  base64data: string;
  width: number;
  height: number;
  timestamp: number;
};

type VideoGridProps = {
  sources: GridSource[];
  selectedSourceId: string | null;
  onSelectSource: (sourceId: string) => void;
  onCloseSource: (sourceId: string) => void;
  onChangeImageRequest: (sourceId: string, sourceNumber: number) => void;
  onCameraFrame: (sourceId: string, payload: CameraFramePayload) => void;
  onCameraStatus: (status: string, type: 'success' | 'error' | 'warning' | 'inactive') => void;
  onCameraError: (title: string, message: string) => void;
};

function getGridTemplate(count: number): { columns: string; rows: string } {
  if (count <= 1) return { columns: '1fr', rows: '1fr' };
  if (count === 2) return { columns: '1fr', rows: 'repeat(2, 1fr)' };
  if (count <= 4) return { columns: 'repeat(2, 1fr)', rows: 'repeat(2, 1fr)' };
  if (count <= 6) return { columns: 'repeat(3, 1fr)', rows: 'repeat(2, 1fr)' };
  return { columns: 'repeat(3, 1fr)', rows: 'repeat(3, 1fr)' };
}

export function VideoGrid({
  sources,
  selectedSourceId,
  onSelectSource,
  onCloseSource,
  onChangeImageRequest,
  onCameraFrame,
  onCameraStatus,
  onCameraError,
}: VideoGridProps) {
  const template = getGridTemplate(sources.length);

  return (
    <div className="react-video-grid-shell">
      <div
        data-testid="video-grid"
        style={{
          display: 'grid',
          gap: 0,
          width: '100%',
          height: '100%',
          gridTemplateColumns: template.columns,
          gridTemplateRows: template.rows,
        }}
      >
        {sources.map((source) => (
          <VideoSourceCard
            key={source.id}
            sourceId={source.id}
            sourceNumber={source.number}
            sourceName={source.name}
            sourceType={source.type}
            imageSrc={source.imageSrc}
            isSelected={selectedSourceId === source.id}
            onSelect={onSelectSource}
            onClose={onCloseSource}
            onChangeImage={onChangeImageRequest}
          >
            {source.type === 'camera' ? (
              <CameraPreview
                onFrameCaptured={(payload) => onCameraFrame(source.id, payload)}
                onCameraStatus={onCameraStatus}
                onCameraError={onCameraError}
              />
            ) : null}
          </VideoSourceCard>
        ))}
      </div>
    </div>
  );
}
