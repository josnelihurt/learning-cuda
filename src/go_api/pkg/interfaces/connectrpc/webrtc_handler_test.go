package connectrpc

import (
	"context"
	"errors"
	"testing"

	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockWebRTCSignalingClient struct {
	stream      pb.WebRTCSignalingService_SignalingStreamClient
	streamError error
}

func (m *mockWebRTCSignalingClient) SignalingStream(ctx context.Context) (pb.WebRTCSignalingService_SignalingStreamClient, error) {
	if m.streamError != nil {
		return nil, m.streamError
	}
	return m.stream, nil
}

func TestNewWebRTCSignalingHandler(t *testing.T) {
	mockClient := &mockWebRTCSignalingClient{}
	handler := NewWebRTCSignalingHandler(mockClient)

	require.NotNil(t, handler)
	assert.Equal(t, mockClient, handler.client)
}

func TestSignalingStream_GrpcError_Propagates(t *testing.T) {
	mockClient := &mockWebRTCSignalingClient{
		streamError: errors.New("grpc connection failed"),
	}
	handler := NewWebRTCSignalingHandler(mockClient)

	require.NotNil(t, handler)
	assert.Equal(t, mockClient, handler.client)
}
