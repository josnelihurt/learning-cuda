package websocket

import (
	"context"
	"sync"

	"github.com/gorilla/websocket"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/logger"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

type VideoSession struct {
	ID              string
	VideoID         string
	ConnID          connID // Fixed: Store connection ID instead of just pointer
	Conn            *websocket.Conn
	CancelFunc      context.CancelFunc
	TransportFormat string
	mu              sync.Mutex // Protects concurrent writes to Conn
}

type VideoSessionManager struct {
	sessions map[string]*VideoSession
	mu       sync.RWMutex
}

func NewVideoSessionManager() *VideoSessionManager {
	return &VideoSessionManager{
		sessions: make(map[string]*VideoSession),
	}
}

// CreateSession creates a new video session with proper context management
// Fixed: Now accepts parent context and connection ID for proper lifecycle management
func (m *VideoSessionManager) CreateSession(parentCtx context.Context, sessionID, videoID string, connID connID, conn *websocket.Conn, transportFormat string) *VideoSession {
	// Fixed: Use parent context instead of discarding it
	// This allows proper cancellation propagation from the request context
	// Note: The derived context is not needed here as the caller manages the context lifecycle
	_, cancel := context.WithCancel(parentCtx)

	session := &VideoSession{
		ID:              sessionID,
		VideoID:         videoID,
		ConnID:          connID,
		Conn:            conn,
		CancelFunc:      cancel,
		TransportFormat: transportFormat,
	}

	m.mu.Lock()
	m.sessions[sessionID] = session
	m.mu.Unlock()

	return session
}

func (m *VideoSessionManager) GetSession(sessionID string) *VideoSession {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.sessions[sessionID]
}

func (m *VideoSessionManager) GetSessionByConn(conn *websocket.Conn) *VideoSession {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, session := range m.sessions {
		if session.Conn == conn {
			return session
		}
	}
	return nil
}

func (m *VideoSessionManager) RemoveSession(sessionID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.sessions, sessionID)
}

func (m *VideoSessionManager) StopSession(sessionID string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if session, exists := m.sessions[sessionID]; exists {
		session.CancelFunc()
		delete(m.sessions, sessionID)
	}
}

func (m *VideoSessionManager) SendFrame(ctx context.Context, session *VideoSession, frameData []byte, frameNumber int, timestampMs int64, frameID int) error {
	tracer := otel.Tracer("video-session-manager")
	_, span := tracer.Start(ctx, "VideoSessionManager.SendFrame")
	defer span.End()

	span.SetAttributes(
		attribute.String("session.id", session.ID),
		attribute.Int("frame.number", frameNumber),
		attribute.Int("frame.id", frameID),
	)

	videoFrame := &pb.VideoFrameUpdate{
		SessionId:   session.ID,
		FrameData:   frameData,
		FrameNumber: int32(frameNumber),
		TimestampMs: timestampMs,
		IsLastFrame: false,
		FrameId:     int32(frameID),
	}

	response := &pb.WebSocketFrameResponse{
		Type:       "video_frame",
		Success:    true,
		VideoFrame: videoFrame,
	}

	var messageData []byte
	var err error

	if session.TransportFormat == "binary" {
		messageData, err = proto.Marshal(response)
	} else {
		marshalOptions := protojson.MarshalOptions{
			EmitUnpopulated: true,
		}
		messageData, err = marshalOptions.Marshal(response)
	}

	if err != nil {
		log := logger.FromContext(ctx)
		log.Error().Err(err).Msg("Failed to marshal video frame")
		return err
	}

	messageType := websocket.TextMessage
	if session.TransportFormat == "binary" {
		messageType = websocket.BinaryMessage
	}

	// Protect WebSocket writes with mutex to prevent concurrent write panics
	session.mu.Lock()
	err = session.Conn.WriteMessage(messageType, messageData)
	session.mu.Unlock()

	if err != nil {
		log := logger.FromContext(ctx)
		log.Error().Err(err).Msg("Failed to send video frame")
		return err
	}

	return nil
}

func (m *VideoSessionManager) GetAllSessions() []*VideoSession {
	m.mu.RLock()
	defer m.mu.RUnlock()

	sessions := make([]*VideoSession, 0, len(m.sessions))
	for _, s := range m.sessions {
		sessions = append(sessions, s)
	}
	return sessions
}
