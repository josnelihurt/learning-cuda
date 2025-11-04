package domain

import (
	"context"

	pb "github.com/jrb/cuda-learning/proto/gen"
)

// ImageProcessor defines the interface for image processing
type ImageProcessor interface {
	ProcessImage(ctx context.Context, img *Image, filters []FilterType, accelerator AcceleratorType, grayscaleType GrayscaleType, blurParams *pb.GaussianBlurParameters) (*Image, error)
}
