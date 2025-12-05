package connectrpc

import (
	"context"
	"errors"
	"fmt"
	"time"

	"connectrpc.com/connect"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/proto/gen/genconnect"
	"github.com/jrb/cuda-learning/webserver/pkg/config"
	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	domainInterfaces "github.com/jrb/cuda-learning/webserver/pkg/domain/interfaces"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/processor"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type RemoteManagementHandler struct {
	grpcClient    *processor.GRPCClient
	config        *config.Manager
	deviceMonitor domainInterfaces.MQTTDeviceMonitor
}

func NewRemoteManagementHandler(grpcClient *processor.GRPCClient, configManager *config.Manager, deviceMonitor domainInterfaces.MQTTDeviceMonitor) *RemoteManagementHandler {
	return &RemoteManagementHandler{
		grpcClient:    grpcClient,
		config:        configManager,
		deviceMonitor: deviceMonitor,
	}
}

func (h *RemoteManagementHandler) StartJetsonNano(
	ctx context.Context,
	req *connect.Request[pb.StartJetsonNanoRequest],
) (*connect.Response[pb.StartJetsonNanoResponse], error) {
	span := trace.SpanFromContext(ctx)

	if h.deviceMonitor == nil {
		span.RecordError(fmt.Errorf("device monitor not available"))
		return nil, connect.NewError(connect.CodeInternal, errors.New("device monitor not initialized"))
	}

	if err := h.deviceMonitor.PowerOn(); err != nil {
		span.RecordError(err)
		return connect.NewResponse(&pb.StartJetsonNanoResponse{
			Status:       pb.StartJetsonNanoStatus_START_JETSON_NANO_STATUS_ERROR,
			Step:         "error",
			Message:      fmt.Sprintf("Failed to send power command: %v", err),
			TraceContext: req.Msg.TraceContext,
		}), nil
	}

	span.SetAttributes(
		attribute.Bool("jetson.power_command_sent", true),
	)

	return connect.NewResponse(&pb.StartJetsonNanoResponse{
		Status:       pb.StartJetsonNanoStatus_START_JETSON_NANO_STATUS_SUCCESS,
		Step:         "sent",
		Message:      "POWER ON command sent successfully",
		TraceContext: req.Msg.TraceContext,
	}), nil
}

type healthCheckResult struct {
	status         pb.AcceleratorHealthStatus
	message        string
	serverVersion  string
	libraryVersion string
}

func (h *RemoteManagementHandler) checkAcceleratorHealth(ctx context.Context) healthCheckResult {
	span := trace.SpanFromContext(ctx)

	if h.grpcClient == nil {
		span.RecordError(fmt.Errorf("grpc client not available"))
		return healthCheckResult{
			status:  pb.AcceleratorHealthStatus_ACCELERATOR_HEALTH_STATUS_UNHEALTHY,
			message: "gRPC client not configured",
		}
	}

	versionReq := &pb.GetVersionInfoRequest{}

	versionResp, err := h.grpcClient.GetVersionInfo(ctx, versionReq)
	if err != nil {
		span.RecordError(err)
		span.SetAttributes(
			attribute.Bool("accelerator.healthy", false),
		)

		return healthCheckResult{
			status:  pb.AcceleratorHealthStatus_ACCELERATOR_HEALTH_STATUS_UNHEALTHY,
			message: fmt.Sprintf("Failed to check accelerator health: %v", err),
		}
	}

	if versionResp.Code != 0 {
		span.SetAttributes(
			attribute.Bool("accelerator.healthy", false),
			attribute.Int("accelerator.error_code", int(versionResp.Code)),
		)

		return healthCheckResult{
			status:  pb.AcceleratorHealthStatus_ACCELERATOR_HEALTH_STATUS_UNHEALTHY,
			message: fmt.Sprintf("Accelerator returned error code %d: %s", versionResp.Code, versionResp.Message),
		}
	}

	span.SetAttributes(
		attribute.Bool("accelerator.healthy", true),
		attribute.String("accelerator.server_version", versionResp.ServerVersion),
		attribute.String("accelerator.library_version", versionResp.LibraryVersion),
	)

	return healthCheckResult{
		status:         pb.AcceleratorHealthStatus_ACCELERATOR_HEALTH_STATUS_HEALTHY,
		message:        "Accelerator is healthy",
		serverVersion:  versionResp.ServerVersion,
		libraryVersion: versionResp.LibraryVersion,
	}
}

func (h *RemoteManagementHandler) CheckAcceleratorHealth(
	ctx context.Context,
	req *connect.Request[pb.CheckAcceleratorHealthRequest],
) (*connect.Response[pb.CheckAcceleratorHealthResponse], error) {
	result := h.checkAcceleratorHealth(ctx)

	return connect.NewResponse(&pb.CheckAcceleratorHealthResponse{
		Status:         result.status,
		Message:        result.message,
		ServerVersion:  result.serverVersion,
		LibraryVersion: result.libraryVersion,
		TraceContext:   req.Msg.TraceContext,
	}), nil
}

func (h *RemoteManagementHandler) MonitorJetsonNano(
	ctx context.Context,
	req *connect.Request[pb.MonitorJetsonNanoRequest],
	stream *connect.ServerStream[pb.MonitorJetsonNanoResponse],
) error {
	span := trace.SpanFromContext(ctx)

	if h.deviceMonitor == nil {
		span.RecordError(fmt.Errorf("device monitor not available"))
		return connect.NewError(connect.CodeInternal, errors.New("device monitor not initialized"))
	}

	initialMsg := "Connected to MQTT device monitor. Sending last known status and recent messages..."
	if err := stream.Send(&pb.MonitorJetsonNanoResponse{
		Data: initialMsg,
	}); err != nil {
		span.RecordError(err)
		return err
	}

	updateChan := make(chan *domain.DeviceStatus, 10)
	healthChan := make(chan healthCheckResult, 10)

	unsubscribe := h.deviceMonitor.Subscribe(func(status *domain.DeviceStatus) {
		select {
		case updateChan <- status:
		default:
		}
	})
	defer unsubscribe()

	go func() {
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				result := h.checkAcceleratorHealth(ctx)
				select {
				case healthChan <- result:
				default:
				}
			}
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case status := <-updateChan:
			if err := stream.Send(&pb.MonitorJetsonNanoResponse{
				Data: status.String(),
			}); err != nil {
				span.RecordError(err)
				return err
			}
		case health := <-healthChan:
			healthMsg := fmt.Sprintf("Accelerator Health: %s - %s", health.status.String(), health.message)
			if health.serverVersion != "" {
				healthMsg += fmt.Sprintf(" (Server: %s, Library: %s)", health.serverVersion, health.libraryVersion)
			}
			if err := stream.Send(&pb.MonitorJetsonNanoResponse{
				Data: healthMsg,
			}); err != nil {
				span.RecordError(err)
				return err
			}
		}
	}
}

var _ genconnect.RemoteManagementServiceHandler = (*RemoteManagementHandler)(nil)
