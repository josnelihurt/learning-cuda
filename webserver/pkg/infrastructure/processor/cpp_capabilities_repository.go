package processor

import (
	"context"
	"fmt"

	pb "github.com/jrb/cuda-learning/proto/gen"
)

type CPPCapabilitiesRepository struct {
	connector *CppConnector
}

func NewCPPCapabilitiesRepository(connector *CppConnector) *CPPCapabilitiesRepository {
	return &CPPCapabilitiesRepository{
		connector: connector,
	}
}

func (r *CPPCapabilitiesRepository) GetCapabilities(_ context.Context) (*pb.LibraryCapabilities, error) {
	if r.connector == nil {
		return nil, fmt.Errorf("cpp connector not initialized")
	}

	caps := r.connector.GetCapabilities()
	if caps == nil {
		return nil, fmt.Errorf("cpp capabilities not available")
	}

	return caps, nil
}
