package connectrpc

import (
	"context"
	"errors"

	"connectrpc.com/connect"
	pb "github.com/jrb/cuda-learning/proto/gen"
	videoapp "github.com/jrb/cuda-learning/src/go_api/pkg/application/media/video"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/logger"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type FileHandler struct {
	listAvailableImagesUseCase listAvailableImagesUseCase
	uploadImageUseCase         uploadImageUseCase
	listAvailableVideosUseCase listVideosUseCase
	uploadVideoUseCase         uploadVideoUseCase
}

func NewFileHandler(
	listAvailableImagesUC listAvailableImagesUseCase,
	uploadImageUC uploadImageUseCase,
	listAvailableVideosUC listVideosUseCase,
	uploadVideoUC uploadVideoUseCase,
) *FileHandler {
	return &FileHandler{
		listAvailableImagesUseCase: listAvailableImagesUC,
		uploadImageUseCase:         uploadImageUC,
		listAvailableVideosUseCase: listAvailableVideosUC,
		uploadVideoUseCase:         uploadVideoUC,
	}
}

func (h *FileHandler) ListAvailableImages(
	ctx context.Context,
	req *connect.Request[pb.ListAvailableImagesRequest],
) (*connect.Response[pb.ListAvailableImagesResponse], error) {
	span := trace.SpanFromContext(ctx)

	images, err := h.listAvailableImagesUseCase.Execute(ctx)
	if err != nil {
		span.RecordError(err)
		logger.FromContext(ctx).Error().Err(err).Msg("Failed to list available images")
		return nil, connect.NewError(connect.CodeInternal, err)
	}

	pbImages := make([]*pb.StaticImage, 0, len(images))
	for _, img := range images {
		pbImages = append(pbImages, &pb.StaticImage{
			Id:          img.ID,
			DisplayName: img.DisplayName,
			Path:        img.Path,
			IsDefault:   img.IsDefault,
		})
	}

	span.SetAttributes(
		attribute.Int("available_images.count", len(pbImages)),
	)

	return connect.NewResponse(&pb.ListAvailableImagesResponse{
		Images: pbImages,
	}), nil
}

func (h *FileHandler) UploadImage(
	ctx context.Context,
	req *connect.Request[pb.UploadImageRequest],
) (*connect.Response[pb.UploadImageResponse], error) {
	span := trace.SpanFromContext(ctx)

	span.SetAttributes(
		attribute.String("filename", req.Msg.Filename),
		attribute.Int("file_size", len(req.Msg.FileData)),
	)

	image, err := h.uploadImageUseCase.Execute(ctx, req.Msg.Filename, req.Msg.FileData)
	if err != nil {
		span.RecordError(err)
		logger.FromContext(ctx).Error().Err(err).Msg("Failed to upload image")

		code := connect.CodeInternal
		if err.Error() == "file too large" {
			code = connect.CodeInvalidArgument
		} else if err.Error() == "invalid format" {
			code = connect.CodeInvalidArgument
		}

		return nil, connect.NewError(code, err)
	}

	span.SetAttributes(
		attribute.String("image.id", image.ID),
		attribute.String("image.path", image.Path),
	)

	logger.FromContext(ctx).Info().Str("image_id", image.ID).Msg("Image uploaded successfully")

	return connect.NewResponse(&pb.UploadImageResponse{
		Image: &pb.StaticImage{
			Id:          image.ID,
			DisplayName: image.DisplayName,
			Path:        image.Path,
			IsDefault:   image.IsDefault,
		},
		Message: "Image uploaded successfully",
	}), nil
}

func (h *FileHandler) ListAvailableVideos(
	ctx context.Context,
	req *connect.Request[pb.ListAvailableVideosRequest],
) (*connect.Response[pb.ListAvailableVideosResponse], error) {
	span := trace.SpanFromContext(ctx)

	videos, err := h.listAvailableVideosUseCase.Execute(ctx)
	if err != nil {
		span.RecordError(err)
		logger.FromContext(ctx).Error().Err(err).Msg("Failed to list available videos")
		return nil, connect.NewError(connect.CodeInternal, err)
	}

	pbVideos := make([]*pb.StaticVideo, 0, len(videos))
	for _, vid := range videos {
		pbVideos = append(pbVideos, &pb.StaticVideo{
			Id:               vid.ID,
			DisplayName:      vid.DisplayName,
			Path:             vid.Path,
			PreviewImagePath: vid.PreviewImagePath,
			IsDefault:        vid.IsDefault,
		})
	}

	span.SetAttributes(
		attribute.Int("available_videos.count", len(pbVideos)),
	)

	return connect.NewResponse(&pb.ListAvailableVideosResponse{
		Videos: pbVideos,
	}), nil
}

func (h *FileHandler) UploadVideo(
	ctx context.Context,
	req *connect.Request[pb.UploadVideoRequest],
) (*connect.Response[pb.UploadVideoResponse], error) {
	span := trace.SpanFromContext(ctx)

	span.SetAttributes(
		attribute.String("filename", req.Msg.Filename),
		attribute.Int("file_size", len(req.Msg.FileData)),
	)

	video, err := h.uploadVideoUseCase.Execute(ctx, req.Msg.FileData, req.Msg.Filename)
	if err != nil {
		span.RecordError(err)
		logger.FromContext(ctx).Error().Err(err).Msg("Failed to upload video")

		code := connect.CodeInternal
		if errors.Is(err, videoapp.ErrFileTooLarge) {
			code = connect.CodeInvalidArgument
		} else if errors.Is(err, videoapp.ErrInvalidFormat) {
			code = connect.CodeInvalidArgument
		}

		return nil, connect.NewError(code, err)
	}

	span.SetAttributes(
		attribute.String("video.id", video.ID),
		attribute.String("video.path", video.Path),
	)

	logger.FromContext(ctx).Info().Str("video_id", video.ID).Msg("Video uploaded successfully")

	return connect.NewResponse(&pb.UploadVideoResponse{
		Video: &pb.StaticVideo{
			Id:               video.ID,
			DisplayName:      video.DisplayName,
			Path:             video.Path,
			PreviewImagePath: video.PreviewImagePath,
			IsDefault:        video.IsDefault,
		},
		Message: "Video uploaded successfully",
	}), nil
}
