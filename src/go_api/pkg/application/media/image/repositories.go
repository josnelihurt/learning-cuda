package image

import (
	"context"

	"github.com/jrb/cuda-learning/src/go_api/pkg/domain"
)

type staticImageRepository interface {
	FindAll(ctx context.Context) ([]domain.StaticImage, error)
	Save(ctx context.Context, filename string, data []byte) (*domain.StaticImage, error)
}

type imageProcessor interface {
	ProcessImage(ctx context.Context, img *domain.Image, opts domain.ProcessingOptions) (*domain.Image, error)
}
