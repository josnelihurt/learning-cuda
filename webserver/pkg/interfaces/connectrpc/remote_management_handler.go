package connectrpc

import (
	"context"
	"errors"
	"fmt"

	"connectrpc.com/connect"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/proto/gen/genconnect"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/processor"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type RemoteManagementHandler struct {
	grpcClient *processor.GRPCClient
}

func NewRemoteManagementHandler(grpcClient *processor.GRPCClient) *RemoteManagementHandler {
	return &RemoteManagementHandler{
		grpcClient: grpcClient,
	}
}

func (h *RemoteManagementHandler) StartJetsonNano(
	ctx context.Context,
	req *connect.Request[pb.StartJetsonNanoRequest],
	stream *connect.ServerStream[pb.StartJetsonNanoResponse],
) error {
	return connect.NewError(connect.CodeUnimplemented, errors.New("remote management not yet implemented"))
}

func (h *RemoteManagementHandler) CheckAcceleratorHealth(
	ctx context.Context,
	req *connect.Request[pb.CheckAcceleratorHealthRequest],
) (*connect.Response[pb.CheckAcceleratorHealthResponse], error) {
	span := trace.SpanFromContext(ctx)

	if h.grpcClient == nil {
		span.RecordError(fmt.Errorf("grpc client not available"))
		return nil, connect.NewError(
			connect.CodeInternal,
			errors.New("gRPC client not configured"),
		)
	}

	versionReq := &pb.GetVersionInfoRequest{
		TraceContext: req.Msg.TraceContext,
	}

	versionResp, err := h.grpcClient.GetVersionInfo(ctx, versionReq)
	if err != nil {
		span.RecordError(err)
		span.SetAttributes(
			attribute.Bool("accelerator.healthy", false),
		)

		return connect.NewResponse(&pb.CheckAcceleratorHealthResponse{
			Status:       pb.AcceleratorHealthStatus_ACCELERATOR_HEALTH_STATUS_UNHEALTHY,
			Message:      fmt.Sprintf("Failed to check accelerator health: %v", err),
			TraceContext: req.Msg.TraceContext,
		}), nil
	}

	if versionResp.Code != 0 {
		span.SetAttributes(
			attribute.Bool("accelerator.healthy", false),
			attribute.Int("accelerator.error_code", int(versionResp.Code)),
		)

		return connect.NewResponse(&pb.CheckAcceleratorHealthResponse{
			Status:       pb.AcceleratorHealthStatus_ACCELERATOR_HEALTH_STATUS_UNHEALTHY,
			Message:      fmt.Sprintf("Accelerator returned error code %d: %s", versionResp.Code, versionResp.Message),
			TraceContext: req.Msg.TraceContext,
		}), nil
	}

	span.SetAttributes(
		attribute.Bool("accelerator.healthy", true),
		attribute.String("accelerator.server_version", versionResp.ServerVersion),
		attribute.String("accelerator.library_version", versionResp.LibraryVersion),
	)

	return connect.NewResponse(&pb.CheckAcceleratorHealthResponse{
		Status:         pb.AcceleratorHealthStatus_ACCELERATOR_HEALTH_STATUS_HEALTHY,
		Message:        "Accelerator is healthy",
		ServerVersion:  versionResp.ServerVersion,
		LibraryVersion: versionResp.LibraryVersion,
		TraceContext:   req.Msg.TraceContext,
	}), nil
}

var _ genconnect.RemoteManagementServiceHandler = (*RemoteManagementHandler)(nil)
