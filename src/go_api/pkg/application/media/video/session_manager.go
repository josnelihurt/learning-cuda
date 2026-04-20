package video

import (
	"errors"
	"sync"
)

var (
	ErrSessionAlreadyExists = errors.New("session already exists")
	ErrSessionNotFound      = errors.New("session not found")
)

type VideoSessionManager struct {
	mu       sync.Mutex
	sessions map[string]*videoPlaybackSession
}

func NewVideoSessionManager() *VideoSessionManager {
	return &VideoSessionManager{
		sessions: make(map[string]*videoPlaybackSession),
	}
}

func (m *VideoSessionManager) CreateSession(sessionID string, session *videoPlaybackSession) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.sessions[sessionID]; exists {
		return ErrSessionAlreadyExists
	}

	m.sessions[sessionID] = session
	return nil
}

func (m *VideoSessionManager) GetSession(sessionID string) (*videoPlaybackSession, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	session, exists := m.sessions[sessionID]
	return session, exists
}

func (m *VideoSessionManager) DeleteSession(sessionID string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.sessions, sessionID)
}

func (m *VideoSessionManager) SessionExists(sessionID string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()

	_, exists := m.sessions[sessionID]
	return exists
}
