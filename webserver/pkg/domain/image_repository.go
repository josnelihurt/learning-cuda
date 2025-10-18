package domain

import "context"

type StaticImage struct {
	ID          string
	DisplayName string
	Path        string
	IsDefault   bool
}

type StaticImageRepository interface {
	FindAll(ctx context.Context) ([]StaticImage, error)
}
