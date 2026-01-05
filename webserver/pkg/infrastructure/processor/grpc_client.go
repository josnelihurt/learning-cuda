package processor

import (
	"context"
	"fmt"
	"time"

	gen "github.com/jrb/cuda-learning/proto/gen"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type GRPCClient struct {
	conn         *grpc.ClientConn
	client       gen.ImageProcessorServiceClient
	webrtcClient gen.WebRTCSignalingServiceClient
}

type GRPCClientConfig struct {
	Address      string
	DialTimeout  time.Duration
	MaxRecvBytes int
	MaxSendBytes int
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
	}, nil
}

func (c *GRPCClient) Close() error {
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

func (c *GRPCClient) ListFilters(ctx context.Context) (*gen.ListFiltersResponse, error) {
	if c.client == nil {
		return nil, fmt.Errorf("grpc client not initialized")
	}

	resp, err := c.client.ListFilters(ctx, &gen.ListFiltersRequest{})
	if err != nil {
		return nil, fmt.Errorf("grpc ListFilters call failed: %w", err)
	}

	return resp, nil
}

func (c *GRPCClient) ProcessImage(ctx context.Context, req *gen.ProcessImageRequest) (*gen.ProcessImageResponse, error) {
	if c.client == nil {
		return nil, fmt.Errorf("grpc client not initialized")
	}

	resp, err := c.client.ProcessImage(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("grpc ProcessImage call failed: %w", err)
	}

	return resp, nil
}

func (c *GRPCClient) GetVersionInfo(ctx context.Context, req *gen.GetVersionInfoRequest) (*gen.GetVersionInfoResponse, error) {
	if c.client == nil {
		return nil, fmt.Errorf("grpc client not initialized")
	}

	resp, err := c.client.GetVersionInfo(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("grpc GetVersionInfo call failed: %w", err)
	}

	return resp, nil
}

func (c *GRPCClient) SignalingStream(ctx context.Context) (gen.WebRTCSignalingService_SignalingStreamClient, error) {
	if c.webrtcClient == nil {
		return nil, fmt.Errorf("webrtc signaling client not initialized")
	}

	stream, err := c.webrtcClient.SignalingStream(ctx)
	if err != nil {
		return nil, fmt.Errorf("grpc SignalingStream call failed: %w", err)
	}

	return stream, nil
}
