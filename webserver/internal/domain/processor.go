package domain

// ImageProcessor defines the interface for image processing
type ImageProcessor interface {
	ProcessImage(img *Image, filter FilterType) (*Image, error)
}

