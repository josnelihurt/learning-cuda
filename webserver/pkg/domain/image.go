package domain

// Image represents an image with its data and metadata
type Image struct {
	Data   []byte
	Width  int
	Height int
	Format string
}

// FilterType represents the type of filter to apply
type FilterType string

const (
	FilterNone      FilterType = "none"
	FilterGrayscale FilterType = "grayscale"
)

// AcceleratorType represents the processing unit to use
type AcceleratorType string

const (
	AcceleratorGPU AcceleratorType = "gpu"
	AcceleratorCPU AcceleratorType = "cpu"
)

// GrayscaleType represents the grayscale conversion algorithm
type GrayscaleType string

const (
	GrayscaleBT601      GrayscaleType = "bt601"      // ITU-R BT.601 (SDTV)
	GrayscaleBT709      GrayscaleType = "bt709"      // ITU-R BT.709 (HDTV)
	GrayscaleAverage    GrayscaleType = "average"    // Simple average
	GrayscaleLightness  GrayscaleType = "lightness"  // (max + min) / 2
	GrayscaleLuminosity GrayscaleType = "luminosity" // Weighted average
)

