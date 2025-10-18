package application

import (
	"context"
	"errors"
	"fmt"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

const maxFileSize = 10 * 1024 * 1024

var (
	errFileTooLarge  = errors.New("file too large")
	errInvalidFormat = errors.New("invalid format")
	errEmptyFilename = errors.New("empty filename")
	errEmptyFileData = errors.New("empty file data")
)

type UploadImageUseCase struct {
	repository domain.StaticImageRepository
}

func NewUploadImageUseCase(repository domain.StaticImageRepository) *UploadImageUseCase {
	return &UploadImageUseCase{
		repository: repository,
	}
}

func (uc *UploadImageUseCase) Execute(ctx context.Context, filename string, fileData []byte) (*domain.StaticImage, error) {
	tracer := otel.Tracer("upload-image")
	ctx, span := tracer.Start(ctx, "UploadImageUseCase.Execute",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(
		attribute.String("filename", filename),
		attribute.Int("file_size", len(fileData)),
	)

	if filename == "" {
		span.RecordError(errEmptyFilename)
		return nil, errEmptyFilename
	}

	if len(fileData) == 0 {
		span.RecordError(errEmptyFileData)
		return nil, errEmptyFileData
	}

	if len(fileData) > maxFileSize {
		span.RecordError(errFileTooLarge)
		span.SetAttributes(attribute.String("validation.error", "file_too_large"))
		return nil, errFileTooLarge
	}

	if !isPNGFormat(fileData) {
		span.RecordError(errInvalidFormat)
		span.SetAttributes(attribute.String("validation.error", "invalid_format"))
		return nil, errInvalidFormat
	}

	image, err := uc.repository.Save(ctx, filename, fileData)
	if err != nil {
		span.RecordError(err)
		return nil, fmt.Errorf("failed to save image: %w", err)
	}

	span.SetAttributes(
		attribute.String("image.id", image.ID),
		attribute.String("image.path", image.Path),
	)

	return image, nil
}

func isPNGFormat(data []byte) bool {
	if len(data) < 8 {
		return false
	}
	pngHeader := []byte{137, 80, 78, 71, 13, 10, 26, 10}
	for i := 0; i < 8; i++ {
		if data[i] != pngHeader[i] {
			return false
		}
	}
	return true
}
