package container

import (
	"context"

	"github.com/jrb/cuda-learning/src/go_api/pkg/application/media/video"
	"github.com/jrb/cuda-learning/src/go_api/pkg/domain"
)

type processImageUseCase interface {
	Execute(ctx context.Context, img *domain.Image, opts domain.ProcessingOptions) (*domain.Image, error)
}

type getSystemInfoUseCase interface {
	Execute(ctx context.Context) (*domain.SystemInfo, error)
}

type listInputsUseCase interface {
	Execute(ctx context.Context) ([]video.InputSource, error)
}

type listAvailableImagesUseCase interface {
	Execute(ctx context.Context) ([]domain.StaticImage, error)
}

type uploadImageUseCase interface {
	Execute(ctx context.Context, filename string, fileData []byte) (*domain.StaticImage, error)
}

type listVideosUseCase interface {
	Execute(ctx context.Context) ([]domain.Video, error)
}

type uploadVideoUseCase interface {
	Execute(ctx context.Context, fileData []byte, filename string) (*domain.Video, error)
}

type evaluateFeatureFlagUseCase interface {
	EvaluateBoolean(ctx context.Context, flagKey string, entityID string, fallbackValue bool) (bool, error)
	EvaluateString(ctx context.Context, flagKey string, entityID string, fallbackValue string) (string, error)
}
