package domain

// ProcessingOptions contains all configuration for image/video processing.
type ProcessingOptions struct {
	Filters       []FilterType
	Accelerator   AcceleratorType
	GrayscaleType GrayscaleType
	BlurParams    *BlurParameters
}
