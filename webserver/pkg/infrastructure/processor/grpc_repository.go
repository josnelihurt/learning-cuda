package processor

import (
	"context"
	"fmt"

	gen "github.com/jrb/cuda-learning/proto/gen"
)

type GRPCRepository struct {
	client *GRPCClient
}

func NewGRPCRepository(client *GRPCClient) *GRPCRepository {
	return &GRPCRepository{
		client: client,
	}
}

func (r *GRPCRepository) GetCapabilities(ctx context.Context) (*gen.LibraryCapabilities, error) {
	if r.client == nil {
		return nil, fmt.Errorf("grpc client not initialized")
	}

	resp, err := r.client.ListFilters(ctx)
	if err != nil {
		return nil, err
	}

	return &gen.LibraryCapabilities{
		Filters:    nil,
		ApiVersion: resp.ApiVersion,
	}, nil
}
