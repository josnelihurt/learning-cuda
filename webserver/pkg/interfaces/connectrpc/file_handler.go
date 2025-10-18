package connectrpc

import (
	"context"
	"log"

	"connectrpc.com/connect"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type FileHandler struct {
	listAvailableImagesUseCase *application.ListAvailableImagesUseCase
	uploadImageUseCase         *application.UploadImageUseCase
}

func NewFileHandler(
	listAvailableImagesUC *application.ListAvailableImagesUseCase,
	uploadImageUC *application.UploadImageUseCase,
) *FileHandler {
	return &FileHandler{
		listAvailableImagesUseCase: listAvailableImagesUC,
		uploadImageUseCase:         uploadImageUC,
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
		log.Printf("Failed to list available images: %v", err)
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
		log.Printf("Failed to upload image: %v", err)

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

	log.Printf("Image uploaded successfully: %s", image.ID)

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
