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

