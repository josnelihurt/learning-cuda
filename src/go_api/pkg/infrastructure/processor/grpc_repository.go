package processor

import (
	"context"
	"fmt"

	gen "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/src/go_api/pkg/interfaces/adapters"
)

type GRPCRepository struct {
	gateway     *AcceleratorGateway
	filterCodec *adapters.FilterCodec
}

func NewGRPCRepository(gateway *AcceleratorGateway) *GRPCRepository {
	return &GRPCRepository{
		gateway:     gateway,
		filterCodec: adapters.NewFilterCodec(),
	}
}

func (r *GRPCRepository) GetCapabilities(ctx context.Context) (*gen.LibraryCapabilities, error) {
	if r.gateway == nil {
		return nil, fmt.Errorf("accelerator gateway not initialized")
	}

	resp, err := r.gateway.ListFilters(ctx)
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
