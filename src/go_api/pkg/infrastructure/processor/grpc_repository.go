package processor

import (
	"context"
	"fmt"

	gen "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/webserver/pkg/interfaces/adapters"
)

type GRPCRepository struct {
	client      *GRPCClient
	filterCodec *adapters.FilterCodec
}

func NewGRPCRepository(client *GRPCClient) *GRPCRepository {
	return &GRPCRepository{
		client:      client,
		filterCodec: adapters.NewFilterCodec(),
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

	// Convert GenericFilterDefinition to FilterDefinition
	filters := make([]*gen.FilterDefinition, 0, len(resp.Filters))
	for _, genericFilter := range resp.Filters {
		if genericFilter == nil {
			continue
		}
		filterDef := r.filterCodec.ToFilterDefinition(genericFilter)
		if filterDef != nil {
			filters = append(filters, filterDef)
		}
	}

	return &gen.LibraryCapabilities{
		Filters:    filters,
		ApiVersion: resp.ApiVersion,
	}, nil
}
