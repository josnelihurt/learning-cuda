package filesystem

import (
	"context"
	"os"
	"path/filepath"
	"strings"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type StaticImageRepository struct {
	directory string
}

func NewStaticImageRepository(directory string) *StaticImageRepository {
	return &StaticImageRepository{
		directory: directory,
	}
}

func (r *StaticImageRepository) FindAll(ctx context.Context) ([]domain.StaticImage, error) {
	tracer := otel.Tracer("static-image-repository")
	_, span := tracer.Start(ctx, "StaticImageRepository.FindAll",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(
		attribute.String("directory", r.directory),
	)

	entries, err := os.ReadDir(r.directory)
	if err != nil {
		span.RecordError(err)
		span.SetAttributes(attribute.String("error.type", "directory_read_failed"))
		return []domain.StaticImage{}, nil
	}

	var images []domain.StaticImage
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		name := entry.Name()
		ext := strings.ToLower(filepath.Ext(name))
		if ext != ".png" && ext != ".jpg" && ext != ".jpeg" {
			continue
		}

		baseName := strings.TrimSuffix(name, ext)
		id := strings.ToLower(baseName)

		displayName := strings.ToUpper(baseName[:1]) + strings.ReplaceAll(baseName[1:], "_", " ")

		isDefault := id == "lena"

		images = append(images, domain.StaticImage{
			ID:          id,
			DisplayName: displayName,
			Path:        filepath.Join(r.directory, name),
			IsDefault:   isDefault,
		})
	}

	span.SetAttributes(
		attribute.Int("images.count", len(images)),
	)

	return images, nil
}

func (r *StaticImageRepository) Save(ctx context.Context, filename string, data []byte) (*domain.StaticImage, error) {
	tracer := otel.Tracer("static-image-repository")
	_, span := tracer.Start(ctx, "StaticImageRepository.Save",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(
		attribute.String("filename", filename),
		attribute.Int("file_size", len(data)),
	)

	filePath := filepath.Join(r.directory, filename)
	err := os.WriteFile(filePath, data, 0600)
	if err != nil {
		span.RecordError(err)
		span.SetAttributes(attribute.String("error.type", "file_write_failed"))
		return nil, err
	}

	baseName := strings.TrimSuffix(filename, filepath.Ext(filename))
	id := strings.ToLower(baseName)
	displayName := strings.ToUpper(baseName[:1]) + strings.ReplaceAll(baseName[1:], "_", " ")

	image := &domain.StaticImage{
		ID:          id,
		DisplayName: displayName,
		Path:        filePath,
		IsDefault:   false,
	}

	span.SetAttributes(
		attribute.String("image.id", image.ID),
		attribute.String("image.path", image.Path),
	)

	return image, nil
}
