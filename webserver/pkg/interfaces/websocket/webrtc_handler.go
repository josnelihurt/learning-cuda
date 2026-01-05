package websocket

import (
	"context"
	"errors"
	"io"
	"net/http"
	"sync"

	"github.com/gorilla/websocket"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/logger"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/processor"
	"github.com/jrb/cuda-learning/webserver/pkg/telemetry"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

type WebRTCSignalingHandler struct {
	grpcClient *processor.GRPCClient
}

func NewWebRTCSignalingHandler(grpcClient *processor.GRPCClient) *WebRTCSignalingHandler {
	return &WebRTCSignalingHandler{
		grpcClient: grpcClient,
	}
}

func getMessageTypeString(msg *pb.SignalingMessage) string {
	if msg == nil {
		return "unknown"
	}
	switch {
	case msg.GetStartSession() != nil:
		return "start_session"
	case msg.GetStartSessionResponse() != nil:
		return "start_session_response"
	case msg.GetIceCandidate() != nil:
		return "ice_candidate"
	case msg.GetIceCandidateResponse() != nil:
		return "ice_candidate_response"
	case msg.GetCloseSession() != nil:
		return "close_session"
	case msg.GetCloseSessionResponse() != nil:
		return "close_session_response"
	case msg.GetKeepAlive() != nil:
		return "keep_alive"
	default:
		return "unknown"
	}
}

func unmarshalSignalingMessage(messageType int, message []byte) (*pb.SignalingMessage, error) {
	var signalingMsg pb.SignalingMessage
	var err error
	if messageType == websocket.BinaryMessage {
		err = proto.Unmarshal(message, &signalingMsg)
	} else {
		unmarshalOptions := protojson.UnmarshalOptions{
			DiscardUnknown: false,
		}
		err = unmarshalOptions.Unmarshal(message, &signalingMsg)
	}
	return &signalingMsg, err
}

func marshalSignalingMessage(msg *pb.SignalingMessage, useBinary bool) (data []byte, messageType int, err error) {
	if useBinary {
		data, err = proto.Marshal(msg)
		messageType = websocket.BinaryMessage
		return
	}
	marshalOptions := protojson.MarshalOptions{
		EmitUnpopulated: true,
	}
	data, err = marshalOptions.Marshal(msg)
	messageType = websocket.TextMessage
	return
}

func (h *WebRTCSignalingHandler) HandleWebRTCSignaling(w http.ResponseWriter, r *http.Request) {
	ctx := telemetry.ExtractFromHTTPHeaders(r.Context(), r.Header)
	tracer := otel.Tracer("webrtc-signaling-websocket")
	ctx, span := tracer.Start(ctx, "WebRTCSignaling.HandleWebSocket")
	defer span.End()

	log := logger.FromContext(ctx)

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Error().Err(err).Msg("WebSocket upgrade failed")
		span.RecordError(err)
		return
	}
	defer conn.Close()

	log.Info().Msg("WebRTC signaling WebSocket connected")
	span.SetAttributes(attribute.String("websocket.connected", "true"))

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	grpcStream, err := h.grpcClient.SignalingStream(ctx)
	if err != nil {
		log.Error().Err(err).Msg("Failed to create gRPC stream to C++ server")
		span.RecordError(err)
		if writeErr := conn.WriteMessage(websocket.TextMessage, []byte(`{"error":"Failed to connect to gRPC server"}`)); writeErr != nil {
			log.Warn().Err(writeErr).Msg("Failed to send error message to WebSocket")
		}
		return
	}
	defer func() {
		if err := grpcStream.CloseSend(); err != nil {
			log.Warn().Err(err).Msg("Failed to close gRPC stream")
		}
	}()

	var wg sync.WaitGroup
	var wsErr error
	var grpcErr error
	var connMutex sync.Mutex

	wg.Add(1)
	go h.forwardWebSocketToGRPC(ctx, conn, grpcStream, cancel, &wsErr, &grpcErr, &wg)

	for {
		msg, err := grpcStream.Recv()
		if err != nil {
			if errors.Is(err, io.EOF) {
				log.Info().Msg("gRPC stream closed by C++ server")
				break
			}
			log.Error().Err(err).Msg("Failed to receive message from gRPC stream")
			grpcErr = err
			cancel()
			break
		}

		messageTypeStr := getMessageTypeString(msg)
		log.Debug().
			Str("message_type", messageTypeStr).
			Msg("Forwarding message from gRPC to WebSocket")

		messageData, messageType, err := marshalSignalingMessage(msg, false)
		if err != nil {
			log.Error().Err(err).Msg("Failed to marshal signaling message")
			continue
		}

		connMutex.Lock()
		err = conn.WriteMessage(messageType, messageData)
		connMutex.Unlock()

		if err != nil {
			log.Error().Err(err).Msg("Failed to send message to WebSocket")
			wsErr = err
			cancel()
			break
		}
	}

	wg.Wait()

	if wsErr != nil {
		span.RecordError(wsErr)
		span.SetAttributes(attribute.String("error.source", "websocket"))
	}
	if grpcErr != nil {
		span.RecordError(grpcErr)
		span.SetAttributes(attribute.String("error.source", "grpc"))
	}

	log.Info().Msg("WebRTC signaling WebSocket connection closed")
}

func (h *WebRTCSignalingHandler) forwardWebSocketToGRPC(
	ctx context.Context,
	conn *websocket.Conn,
	grpcStream interface {
		Send(*pb.SignalingMessage) error
	},
	cancel context.CancelFunc,
	//nolint:gocritic // ptrToRefParam: pointers needed to set errors from goroutine
	wsErr *error,
	//nolint:gocritic // ptrToRefParam: pointers needed to set errors from goroutine
	grpcErr *error,
	wg *sync.WaitGroup,
) {
	defer wg.Done()
	log := logger.FromContext(ctx)
	for {
		messageType, message, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Error().Err(err).Msg("WebSocket read error")
				*wsErr = err
			} else {
				log.Info().Msg("WebSocket closed by client")
			}
			cancel()
			return
		}

		signalingMsg, err := unmarshalSignalingMessage(messageType, message)
		if err != nil {
			log.Error().Err(err).Msg("Failed to unmarshal signaling message from WebSocket")
			continue
		}

		messageTypeStr := getMessageTypeString(signalingMsg)
		log.Debug().
			Str("message_type", messageTypeStr).
			Msg("Forwarding message from WebSocket to gRPC")

		if err := grpcStream.Send(signalingMsg); err != nil {
			log.Error().Err(err).Msg("Failed to send message to gRPC stream")
			*grpcErr = err
			cancel()
			return
		}
	}
}
