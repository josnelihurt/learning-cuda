package connectrpc

import (
	"context"
	"errors"
	"strings"

	"connectrpc.com/connect"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/proto/gen/genconnect"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/logger"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
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
	logger.Global().Info().
		Str("session_id", req.Msg.SessionId).
		Str("sdp_offer", req.Msg.SdpOffer).
		Msg("WebRTC StartSession request received")
	resp, err := h.client.StartWebRTCSession(ctx, req.Msg)
	if err != nil {
		logger.Global().Error().
			Err(err).
			Str("session_id", req.Msg.SessionId).
			Msg("WebRTC StartSession failed")
	} else {
		logger.Global().Info().
			Str("session_id", req.Msg.SessionId).
			Str("sdp_answer", resp.SdpAnswer).
			Msg("WebRTC StartSession succeeded")
	}
	if err != nil {
		// Check if error is gRPC Unimplemented status
		if st, ok := status.FromError(err); ok && st.Code() == codes.Unimplemented {
			return nil, connect.NewError(connect.CodeUnimplemented, errors.New("WebRTC signaling is not implemented in the gRPC processor server"))
		}
		// Check if error message contains "Unimplemented"
		if strings.Contains(err.Error(), "Unimplemented") {
			return nil, connect.NewError(connect.CodeUnimplemented, errors.New("WebRTC signaling is not implemented in the gRPC processor server"))
		}
		return nil, connect.NewError(connect.CodeInternal, err)
	}

	return connect.NewResponse(resp), nil
}

func (h *WebRTCSignalingHandler) SendIceCandidate(
	ctx context.Context,
	req *connect.Request[pb.SendIceCandidateRequest],
) (*connect.Response[pb.SendIceCandidateResponse], error) {
	logger.Global().Info().
		Str("session_id", req.Msg.SessionId).
		Str("candidate", req.Msg.Candidate.Candidate).
		Str("sdp_mid", req.Msg.Candidate.SdpMid).
		Int32("sdp_mline_index", req.Msg.Candidate.SdpMlineIndex).
		Msg("WebRTC SendIceCandidate request received")
	resp, err := h.client.SendIceCandidate(ctx, req.Msg)
	if err != nil {
		logger.Global().Error().
			Err(err).
			Str("session_id", req.Msg.SessionId).
			Msg("WebRTC SendIceCandidate failed")
	} else {
		logger.Global().Info().
			Str("session_id", req.Msg.SessionId).
			Msg("WebRTC SendIceCandidate succeeded")
	}
	if err != nil {
		// Check if error is gRPC Unimplemented status
		if st, ok := status.FromError(err); ok && st.Code() == codes.Unimplemented {
			return nil, connect.NewError(connect.CodeUnimplemented, errors.New("WebRTC signaling is not implemented in the gRPC processor server"))
		}
		// Check if error message contains "Unimplemented"
		if strings.Contains(err.Error(), "Unimplemented") {
			return nil, connect.NewError(connect.CodeUnimplemented, errors.New("WebRTC signaling is not implemented in the gRPC processor server"))
		}
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
		// Check if error is gRPC Unimplemented status
		if st, ok := status.FromError(err); ok && st.Code() == codes.Unimplemented {
			return nil, connect.NewError(connect.CodeUnimplemented, errors.New("WebRTC signaling is not implemented in the gRPC processor server"))
		}
		// Check if error message contains "Unimplemented"
		if strings.Contains(err.Error(), "Unimplemented") {
			return nil, connect.NewError(connect.CodeUnimplemented, errors.New("WebRTC signaling is not implemented in the gRPC processor server"))
		}
		return nil, connect.NewError(connect.CodeInternal, err)
	}

	return connect.NewResponse(resp), nil
}

var _ genconnect.WebRTCSignalingServiceHandler = (*WebRTCSignalingHandler)(nil)
