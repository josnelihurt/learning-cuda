package domain

import "context"

type Video struct {
	ID               string
	DisplayName      string
	Path             string
	PreviewImagePath string
	IsDefault        bool
}

type VideoRepository interface {
	List(ctx context.Context) ([]Video, error)
	GetByID(ctx context.Context, id string) (*Video, error)
	Save(ctx context.Context, video *Video) error
}
