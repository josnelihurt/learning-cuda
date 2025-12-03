package connectrpc

import (
	"context"

	"connectrpc.com/connect"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/proto/gen/genconnect"
)

type WebRTCSignalingClient interface {
	StartWebRTCSession(ctx context.Context, req *pb.StartSessionRequest) (*pb.StartSessionResponse, error)
	SendIceCandidate(ctx context.Context, req *pb.SendIceCandidateRequest) (*pb.SendIceCandidateResponse, error)
	CloseWebRTCSession(ctx context.Context, req *pb.CloseSessionRequest) (*pb.CloseSessionResponse, error)
}

type WebRTCSignalingHandler struct {
	client WebRTCSignalingClient
}

func NewWebRTCSignalingHandler(client WebRTCSignalingClient) *WebRTCSignalingHandler {
	return &WebRTCSignalingHandler{
		client: client,
	}
}

func (h *WebRTCSignalingHandler) StartSession(
	ctx context.Context,
	req *connect.Request[pb.StartSessionRequest],
) (*connect.Response[pb.StartSessionResponse], error) {
	resp, err := h.client.StartWebRTCSession(ctx, req.Msg)
	if err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}

	return connect.NewResponse(resp), nil
}

func (h *WebRTCSignalingHandler) SendIceCandidate(
	ctx context.Context,
	req *connect.Request[pb.SendIceCandidateRequest],
) (*connect.Response[pb.SendIceCandidateResponse], error) {
	resp, err := h.client.SendIceCandidate(ctx, req.Msg)
	if err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}

	return connect.NewResponse(resp), nil
}

func (h *WebRTCSignalingHandler) CloseSession(
	ctx context.Context,
	req *connect.Request[pb.CloseSessionRequest],
) (*connect.Response[pb.CloseSessionResponse], error) {
	resp, err := h.client.CloseWebRTCSession(ctx, req.Msg)
	if err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}

	return connect.NewResponse(resp), nil
}

var _ genconnect.WebRTCSignalingServiceHandler = (*WebRTCSignalingHandler)(nil)
