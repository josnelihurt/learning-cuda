package domain

// ImageProcessor defines the interface for image processing
type ImageProcessor interface {
	ProcessImage(img *Image, filters []FilterType, accelerator AcceleratorType, grayscaleType GrayscaleType) (*Image, error)
}

