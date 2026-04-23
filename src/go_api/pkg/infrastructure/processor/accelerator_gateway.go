package processor

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	gen "github.com/jrb/cuda-learning/proto/gen"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type AcceleratorGateway struct {
	registry *Registry
}

type AcceleratorGatewayConfig struct {
	Registry *Registry
}

func NewAcceleratorGateway(cfg AcceleratorGatewayConfig) *AcceleratorGateway {
	return &AcceleratorGateway{registry: cfg.Registry}
}

// callAccelerator sends a request envelope and awaits the matching response.
func (c *AcceleratorGateway) callAccelerator(
	ctx context.Context,
	buildPayload func(commandID string) *gen.AcceleratorMessage,
	extractResponse func(resp *gen.AcceleratorMessage) (any, error),
) (any, error) {
	if c.registry == nil {
		return nil, status.Error(codes.Unavailable, "no accelerator registry configured")
	}
	sess, ok := c.registry.First()
	if !ok {
		return nil, status.Error(codes.Unavailable, "no accelerator registered")
	}

	commandID := uuid.NewString()
	msg := buildPayload(commandID)
	msg.CommandId = commandID

	if err := sess.Send(msg); err != nil {
		return nil, fmt.Errorf("send to accelerator: %w", err)
	}

	resp, err := sess.Await(ctx, commandID)
	if err != nil {
		return nil, fmt.Errorf("await response: %w", err)
	}
	if errPayload, ok := resp.GetPayload().(*gen.AcceleratorMessage_Error); ok {
		return nil, fmt.Errorf("accelerator error %s: %s", errPayload.Error.Code, errPayload.Error.Message)
	}
	return extractResponse(resp)
}

func (c *AcceleratorGateway) ListFilters(ctx context.Context) (*gen.ListFiltersResponse, error) {
	result, err := c.callAccelerator(ctx,
		func(commandID string) *gen.AcceleratorMessage {
			return &gen.AcceleratorMessage{
				Payload: &gen.AcceleratorMessage_ListFiltersRequest{
					ListFiltersRequest: &gen.ListFiltersRequest{},
				},
			}
		},
		func(resp *gen.AcceleratorMessage) (any, error) {
			r, ok := resp.GetPayload().(*gen.AcceleratorMessage_ListFiltersResponse)
			if !ok {
				return nil, fmt.Errorf("unexpected response type for ListFilters")
			}
			return r.ListFiltersResponse, nil
		},
	)
	if err != nil {
		return nil, err
	}
	return result.(*gen.ListFiltersResponse), nil
}

func (c *AcceleratorGateway) GetVersionInfo(ctx context.Context, req *gen.GetVersionInfoRequest) (*gen.GetVersionInfoResponse, error) {
	result, err := c.callAccelerator(ctx,
		func(commandID string) *gen.AcceleratorMessage {
			return &gen.AcceleratorMessage{
				Payload: &gen.AcceleratorMessage_GetVersionRequest{
					GetVersionRequest: req,
				},
			}
		},
		func(resp *gen.AcceleratorMessage) (any, error) {
			r, ok := resp.GetPayload().(*gen.AcceleratorMessage_GetVersionResponse)
			if !ok {
				return nil, fmt.Errorf("unexpected response type for GetVersionInfo")
			}
			return r.GetVersionResponse, nil
		},
	)
	if err != nil {
		return nil, err
	}
	return result.(*gen.GetVersionInfoResponse), nil
}

// SignalingStream returns a bidi-stream adapter that routes signaling messages
// through the registered accelerator's control stream instead of a direct gRPC dial.
func (c *AcceleratorGateway) SignalingStream(ctx context.Context) (gen.WebRTCSignalingService_SignalingStreamClient, error) {
	if c.registry == nil {
		return nil, status.Error(codes.Unavailable, "no accelerator registry configured")
	}
	sess, ok := c.registry.First()
	if !ok {
		return nil, status.Error(codes.Unavailable, "no accelerator registered")
	}
	return newSignalingStreamAdapter(ctx, sess), nil
}
