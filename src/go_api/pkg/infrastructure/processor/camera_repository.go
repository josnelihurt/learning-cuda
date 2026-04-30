package processor

import (
	"context"

	videoapp "github.com/jrb/cuda-learning/src/go_api/pkg/application/media/video"
)

// RegistryCameraRepository implements video.cameraRepository by reading the
// Cameras list from the first registered accelerator session.
type RegistryCameraRepository struct {
	registry *Registry
}

func NewRegistryCameraRepository(registry *Registry) *RegistryCameraRepository {
	return &RegistryCameraRepository{registry: registry}
}

func (r *RegistryCameraRepository) ListCameras(ctx context.Context) ([]videoapp.RemoteCamera, error) {
	sess, ok := r.registry.First()
	if !ok || sess == nil {
		return nil, nil
	}
	result := make([]videoapp.RemoteCamera, 0, len(sess.Cameras))
	for _, cam := range sess.Cameras {
		if cam == nil {
			continue
		}
		result = append(result, videoapp.RemoteCamera{
			SensorID:    cam.SensorId,
			DisplayName: cam.DisplayName,
			Model:       cam.Model,
		})
	}
	return result, nil
}
