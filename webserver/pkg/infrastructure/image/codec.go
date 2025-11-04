package image

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
)

type Codec struct{}

func NewImageCodec() *Codec {
	return &Codec{}
}

func (c *Codec) DecodeToRGB(data []byte) (*domain.Image, error) {
	img, err := png.Decode(bytes.NewReader(data))
	if err != nil {
		img, err = jpeg.Decode(bytes.NewReader(data))
		if err != nil {
			return nil, fmt.Errorf("failed to decode image: %w", err)
		}
	}

	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	// Convert to RGB (3 channels) by extracting R, G, B and discarding alpha
	rgbData := make([]byte, width*height*3)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			idx := (y*width + x) * 3
			rgbData[idx] = byte(r >> 8)
			rgbData[idx+1] = byte(g >> 8)
			rgbData[idx+2] = byte(b >> 8)
		}
	}

	return &domain.Image{
		Data:   rgbData,
		Width:  width,
		Height: height,
		Format: "rgb",
	}, nil
}

func (c *Codec) EncodeToPNG(img *domain.Image, isGrayscale bool) ([]byte, error) {
	var buf bytes.Buffer

	if !isGrayscale {
		// Calculate expected data size for RGB (3 channels)
		expectedSize := img.Width * img.Height * 3
		if len(img.Data) < expectedSize {
			return nil, fmt.Errorf("insufficient image data: expected %d bytes, got %d bytes", expectedSize, len(img.Data))
		}
		// Use only the expected size in case there's extra data
		rgbData := img.Data[:expectedSize]
		// Convert RGB to RGBA for PNG encoding (PNG requires RGBA)
		rgbaData := make([]byte, img.Width*img.Height*4)
		for i := 0; i < img.Width*img.Height; i++ {
			rgbaData[i*4] = rgbData[i*3]
			rgbaData[i*4+1] = rgbData[i*3+1]
			rgbaData[i*4+2] = rgbData[i*3+2]
			rgbaData[i*4+3] = 255 // Full opacity for alpha
		}
		resultImg := image.NewRGBA(image.Rect(0, 0, img.Width, img.Height))
		resultImg.Pix = rgbaData
		if err := png.Encode(&buf, resultImg); err != nil {
			return nil, fmt.Errorf("failed to encode RGB image: %w", err)
		}
	} else {
		// Calculate expected data size for grayscale (1 channel)
		expectedSize := img.Width * img.Height
		if len(img.Data) < expectedSize {
			return nil, fmt.Errorf("insufficient image data: expected %d bytes, got %d bytes", expectedSize, len(img.Data))
		}
		// Use only the expected size in case there's extra data
		grayData := img.Data[:expectedSize]
		grayImg := image.NewGray(image.Rect(0, 0, img.Width, img.Height))
		grayImg.Pix = grayData
		if err := png.Encode(&buf, grayImg); err != nil {
			return nil, fmt.Errorf("failed to encode grayscale image: %w", err)
		}
	}

	return buf.Bytes(), nil
}
