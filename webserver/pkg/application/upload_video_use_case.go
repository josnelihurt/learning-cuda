package application

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	videoinfra "github.com/jrb/cuda-learning/webserver/pkg/infrastructure/video"
	"github.com/rs/zerolog/log"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

var (
	ErrInvalidFormat = errors.New("invalid format, only MP4 is supported")
	ErrFileTooLarge  = errors.New("file too large, maximum size is 100MB")
)

const maxVideoSize = 100 * 1024 * 1024

type UploadVideoUseCase struct {
	repository  domain.VideoRepository
	videosDir   string
	previewsDir string
}

func NewUploadVideoUseCase(repository domain.VideoRepository, videosDir, previewsDir string) *UploadVideoUseCase {
	return &UploadVideoUseCase{
		repository:  repository,
		videosDir:   videosDir,
		previewsDir: previewsDir,
	}
}

func (uc *UploadVideoUseCase) Execute(ctx context.Context, fileData []byte, filename string) (*domain.Video, error) {
	tracer := otel.Tracer("upload-video")
	_, span := tracer.Start(ctx, "UploadVideo",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(
		attribute.String("filename", filename),
		attribute.Int("file_size", len(fileData)),
	)

	if err := uc.validateFormat(filename); err != nil {
		span.SetAttributes(attribute.Bool("error.invalid_format", true))
		return nil, err
	}

	if err := uc.validateSize(fileData); err != nil {
		span.SetAttributes(attribute.Bool("error.file_too_large", true))
		return nil, err
	}

	id := strings.TrimSuffix(filename, filepath.Ext(filename))
	videoPath := filepath.Join(uc.videosDir, filename)
	previewPath := filepath.Join(uc.previewsDir, id+".png")

	if err := os.WriteFile(videoPath, fileData, 0600); err != nil {
		span.SetAttributes(attribute.Bool("error", true))
		return nil, fmt.Errorf("failed to save video: %w", err)
	}

	previewImagePath := ""
	if err := videoinfra.GeneratePreview(ctx, videoPath, previewPath); err != nil {
		log.Warn().Err(err).Str("video_id", id).Msg("Failed to generate preview for uploaded video")
		span.AddEvent("preview_generation_failed")
		span.SetAttributes(attribute.String("preview.error", err.Error()))
	} else {
		previewImagePath = filepath.Join("/data/video_previews", id+".png")
		span.SetAttributes(attribute.Bool("preview.generated", true))
	}

	video := &domain.Video{
		ID:               id,
		DisplayName:      strings.ReplaceAll(id, "-", " "),
		Path:             filepath.Join("/data/videos", filename),
		PreviewImagePath: previewImagePath,
		IsDefault:        false,
	}

	if err := uc.repository.Save(ctx, video); err != nil {
		span.SetAttributes(attribute.Bool("error", true))
		return nil, err
	}

	span.SetAttributes(
		attribute.String("video.id", video.ID),
		attribute.Bool("upload.success", true),
	)

	return video, nil
}

func (uc *UploadVideoUseCase) validateFormat(filename string) error {
	ext := strings.ToLower(filepath.Ext(filename))
	if ext != ".mp4" {
		return ErrInvalidFormat
	}
	return nil
}

func (uc *UploadVideoUseCase) validateSize(fileData []byte) error {
	if len(fileData) > maxVideoSize {
		return ErrFileTooLarge
	}
	return nil
}
