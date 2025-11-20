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
	conn   *grpc.ClientConn
	client gen.ImageProcessorServiceClient
}

type GRPCClientConfig struct {
	Address      string
	DialTimeout  time.Duration
	MaxRecvBytes int
	MaxSendBytes int
}

func NewGRPCClient(cfg GRPCClientConfig) (*GRPCClient, error) {
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

	ctx, cancel := context.WithTimeout(context.Background(), cfg.DialTimeout)
	defer cancel()

	conn, err := grpc.DialContext(
		ctx,
		cfg.Address,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(cfg.MaxRecvBytes),
			grpc.MaxCallSendMsgSize(cfg.MaxSendBytes),
		),
		grpc.WithBlock(),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to dial grpc server: %w", err)
	}

	return &GRPCClient{
		conn:   conn,
		client: gen.NewImageProcessorServiceClient(conn),
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
