package image

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
)

type ImageCodec struct{}

func NewImageCodec() *ImageCodec {
	return &ImageCodec{}
}

func (c *ImageCodec) DecodeToRGBA(data []byte) (*domain.Image, error) {
	img, err := png.Decode(bytes.NewReader(data))
	if err != nil {
		img, err = jpeg.Decode(bytes.NewReader(data))
		if err != nil {
			return nil, fmt.Errorf("failed to decode image: %w", err)
		}
	}

	bounds := img.Bounds()
	rgba := image.NewRGBA(bounds)
	for y := 0; y < bounds.Dy(); y++ {
		for x := 0; x < bounds.Dx(); x++ {
			rgba.Set(x, y, img.At(x, y))
		}
	}

	return &domain.Image{
		Data:   rgba.Pix,
		Width:  bounds.Dx(),
		Height: bounds.Dy(),
		Format: "rgba",
	}, nil
}

func (c *ImageCodec) EncodeToPNG(img *domain.Image, isGrayscale bool) ([]byte, error) {
	var buf bytes.Buffer
	
	if !isGrayscale {
		resultImg := image.NewRGBA(image.Rect(0, 0, img.Width, img.Height))
		resultImg.Pix = img.Data
		if err := png.Encode(&buf, resultImg); err != nil {
			return nil, fmt.Errorf("failed to encode RGBA image: %w", err)
		}
	} else {
		grayImg := image.NewGray(image.Rect(0, 0, img.Width, img.Height))
		grayImg.Pix = img.Data
		if err := png.Encode(&buf, grayImg); err != nil {
			return nil, fmt.Errorf("failed to encode grayscale image: %w", err)
		}
	}

	return buf.Bytes(), nil
}

