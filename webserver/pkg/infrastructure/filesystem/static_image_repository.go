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
	ctx, span := tracer.Start(ctx, "StaticImageRepository.FindAll",
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

		displayName := strings.Title(strings.ReplaceAll(baseName, "_", " "))

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
