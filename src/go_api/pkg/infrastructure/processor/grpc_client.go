package processor

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	gen "github.com/jrb/cuda-learning/proto/gen"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

type GRPCClient struct {
	conn         *grpc.ClientConn
	client       gen.ImageProcessorServiceClient
	webrtcClient gen.WebRTCSignalingServiceClient
	registry     *Registry
}

type GRPCClientConfig struct {
	Address      string
	DialTimeout  time.Duration
	MaxRecvBytes int
	MaxSendBytes int
	Registry     *Registry
}

func NewGRPCClient(ctx context.Context, cfg GRPCClientConfig) (*GRPCClient, error) {
	if cfg.Address == "" {
		return nil, fmt.Errorf("grpc address is empty")
	}

	if cfg.DialTimeout <= 0 {
		cfg.DialTimeout = 5 * time.Second
	}

	if cfg.MaxRecvBytes <= 0 {
		cfg.MaxRecvBytes = 64 * 1024 * 1024
	}

	if cfg.MaxSendBytes <= 0 {
		cfg.MaxSendBytes = 64 * 1024 * 1024
	}

	conn, err := grpc.NewClient(
		cfg.Address,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(cfg.MaxRecvBytes),
			grpc.MaxCallSendMsgSize(cfg.MaxSendBytes),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to dial grpc server: %w", err)
	}

	return &GRPCClient{
		conn:         conn,
		client:       gen.NewImageProcessorServiceClient(conn),
		webrtcClient: gen.NewWebRTCSignalingServiceClient(conn),
		registry:     cfg.Registry,
	}, nil
}

func (c *GRPCClient) Close() error {
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

// callAccelerator sends a request envelope and awaits the matching response.
func (c *GRPCClient) callAccelerator(
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

func (c *GRPCClient) ListFilters(ctx context.Context) (*gen.ListFiltersResponse, error) {
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

func (c *GRPCClient) ProcessImage(ctx context.Context, req *gen.ProcessImageRequest) (*gen.ProcessImageResponse, error) {
	result, err := c.callAccelerator(ctx,
		func(commandID string) *gen.AcceleratorMessage {
			return &gen.AcceleratorMessage{
				Payload: &gen.AcceleratorMessage_ProcessImageRequest{
					ProcessImageRequest: req,
				},
			}
		},
		func(resp *gen.AcceleratorMessage) (any, error) {
			r, ok := resp.GetPayload().(*gen.AcceleratorMessage_ProcessImageResponse)
			if !ok {
				return nil, fmt.Errorf("unexpected response type for ProcessImage")
			}
			return r.ProcessImageResponse, nil
		},
	)
	if err != nil {
		return nil, err
	}
	return result.(*gen.ProcessImageResponse), nil
}

func (c *GRPCClient) GetVersionInfo(ctx context.Context, req *gen.GetVersionInfoRequest) (*gen.GetVersionInfoResponse, error) {
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
func (c *GRPCClient) SignalingStream(ctx context.Context) (gen.WebRTCSignalingService_SignalingStreamClient, error) {
	if c.registry == nil {
		return nil, status.Error(codes.Unavailable, "no accelerator registry configured")
	}
	sess, ok := c.registry.First()
	if !ok {
		return nil, status.Error(codes.Unavailable, "no accelerator registered")
	}
	return newSignalingStreamAdapter(ctx, sess), nil
}
