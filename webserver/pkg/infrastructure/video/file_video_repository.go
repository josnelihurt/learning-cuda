package video

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
)

const (
	dataDir        = "data"
	videosSubDir   = "videos"
	previewsSubDir = "video_previews"
)

var rootPath = "/"

type FileVideoRepository struct {
	videosDir      string
	previewsDir    string
	defaultVideoID string
}

func NewFileVideoRepository(ctx context.Context, videosDir, previewsDir string) *FileVideoRepository {
	repo := &FileVideoRepository{
		videosDir:      videosDir,
		previewsDir:    previewsDir,
		defaultVideoID: "sample",
	}

	if err := os.MkdirAll(videosDir, 0o755); err == nil {
		if err := os.MkdirAll(previewsDir, 0o755); err == nil {
			go repo.generatePreviewsForExistingVideos(context.Background()) //nolint:contextcheck
		}
	}

	return repo
}

func (r *FileVideoRepository) List(ctx context.Context) ([]domain.Video, error) {
	tracer := otel.Tracer("video-repository")
	_, span := tracer.Start(ctx, "FileVideoRepository.List")
	defer span.End()

	entries, err := os.ReadDir(r.videosDir)
	if err != nil {
		if os.IsNotExist(err) {
			span.SetAttributes(attribute.Int("videos.count", 0))
			return []domain.Video{}, nil
		}
		return nil, fmt.Errorf("failed to read videos directory: %w", err)
	}

	videos := make([]domain.Video, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		name := entry.Name()
		ext := strings.ToLower(filepath.Ext(name))

		if ext != ".mp4" {
			continue
		}

		id := strings.TrimSuffix(name, ext)
		displayName := strings.ReplaceAll(id, "-", " ")
		if displayName != "" {
			displayName = strings.ToUpper(displayName[:1]) + displayName[1:]
		}

		previewPath := filepath.Join(rootPath, dataDir, previewsSubDir, id+".png")
		previewFsPath := filepath.Join(r.previewsDir, id+".png")

		if _, err := os.Stat(previewFsPath); os.IsNotExist(err) {
			videoFsPath := filepath.Join(r.videosDir, name)
			if err := GeneratePreview(ctx, videoFsPath, previewFsPath); err != nil {
				span.AddEvent("preview_generation_failed")
				span.SetAttributes(
					attribute.String("video.id", id),
					attribute.String("error", err.Error()),
				)
				previewPath = ""
			}
		} else {
			if stat, err := os.Stat(previewFsPath); err == nil && stat.Size() == 0 {
				previewPath = ""
			}
		}

		video := domain.Video{
			ID:               id,
			DisplayName:      displayName,
			Path:             filepath.Join(rootPath, dataDir, videosSubDir, name),
			PreviewImagePath: previewPath,
			IsDefault:        id == r.defaultVideoID,
		}

		videos = append(videos, video)
	}

	span.SetAttributes(attribute.Int("videos.count", len(videos)))
	return videos, nil
}

func (r *FileVideoRepository) GetByID(ctx context.Context, id string) (*domain.Video, error) {
	tracer := otel.Tracer("video-repository")
	_, span := tracer.Start(ctx, "FileVideoRepository.GetByID")
	defer span.End()

	span.SetAttributes(attribute.String("video.id", id))

	videos, err := r.List(ctx)
	if err != nil {
		return nil, err
	}

	for _, video := range videos {
		if video.ID == id {
			return &video, nil
		}
	}

	return nil, fmt.Errorf("video not found: %s", id)
}

func (r *FileVideoRepository) Save(ctx context.Context, video *domain.Video) error {
	tracer := otel.Tracer("video-repository")
	_, span := tracer.Start(ctx, "FileVideoRepository.Save")
	defer span.End()

	span.SetAttributes(
		attribute.String("video.id", video.ID),
		attribute.String("video.path", video.Path),
	)

	return nil
}

func (r *FileVideoRepository) generatePreviewsForExistingVideos(ctx context.Context) {
	entries, err := os.ReadDir(r.videosDir)
	if err != nil {
		return
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		name := entry.Name()
		ext := strings.ToLower(filepath.Ext(name))

		if ext != ".mp4" {
			continue
		}

		id := strings.TrimSuffix(name, ext)
		previewPath := filepath.Join(r.previewsDir, id+".png")
		videoPath := filepath.Join(r.videosDir, name)

		if _, err := os.Stat(previewPath); os.IsNotExist(err) {
			if err := GeneratePreview(ctx, videoPath, previewPath); err != nil {
				otel.GetTracerProvider().Tracer("video-repository").Start(ctx, "generatePreview.error")
			}
		}
	}
}
