package connectrpc

import (
	"context"

	pb "github.com/jrb/cuda-learning/proto/gen"
	systemapp "github.com/jrb/cuda-learning/src/go_api/pkg/application/platform/system"
	"github.com/jrb/cuda-learning/src/go_api/pkg/domain"
)

type processImageUseCase interface {
	Execute(ctx context.Context, img *domain.Image, opts domain.ProcessingOptions) (*domain.Image, error)
}
type listAvailableImagesUseCase interface {
	Execute(ctx context.Context) ([]domain.StaticImage, error)
}
type uploadImageUseCase interface {
	Execute(ctx context.Context, filename string, fileData []byte) (*domain.StaticImage, error)
}
type listVideosUseCase interface {
	Execute(ctx context.Context) ([]domain.Video, error)
}
type uploadVideoUseCase interface {
	Execute(ctx context.Context, fileData []byte, filename string) (*domain.Video, error)
}
type streamVideoUseCase interface {
	Start(ctx context.Context, req *pb.StartVideoPlaybackRequest) (*pb.StartVideoPlaybackResponse, error)
	Stop(ctx context.Context, req *pb.StopVideoPlaybackRequest) (*pb.StopVideoPlaybackResponse, error)
}
type processorCapabilitiesUseCase interface {
	Execute(ctx context.Context, useGRPC bool) (*pb.LibraryCapabilities, systemapp.ProcessorBackendOrigin, error)
}
