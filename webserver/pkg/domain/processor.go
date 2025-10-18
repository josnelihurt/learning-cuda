package domain

import "context"

// ImageProcessor defines the interface for image processing
type ImageProcessor interface {
	ProcessImage(ctx context.Context, img *Image, filters []FilterType, accelerator AcceleratorType, grayscaleType GrayscaleType) (*Image, error)
}
