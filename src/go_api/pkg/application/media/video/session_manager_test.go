package video

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

type MockPeer struct {
	mock.Mock
}

func (m *MockPeer) Connect(ctx context.Context) error {
	args := m.Called(ctx)
	return args.Error(0)
}

func (m *MockPeer) Send(payload []byte) error {
	args := m.Called(payload)
	return args.Error(0)
}

func (m *MockPeer) Close() error {
	args := m.Called()
	return args.Error(0)
}

func (m *MockPeer) Label() string {
	args := m.Called()
	return args.String(0)
}

func TestVideoSessionManager_CreateSession(t *testing.T) {
	// Arrange
	manager := NewVideoSessionManager()
	_, cancel := context.WithCancel(context.Background())
	defer cancel()

	mockPeer := &MockPeer{}
	session := &videoPlaybackSession{
		cancel: cancel,
		done:   make(chan error, 1),
		peer:   mockPeer,
	}

	// Act
	err := manager.CreateSession("test-session", session)

	// Assert
	assert.NoError(t, err)
	assert.True(t, manager.SessionExists("test-session"))
}

func TestVideoSessionManager_CreateSession_AlreadyExists(t *testing.T) {
	// Arrange
	manager := NewVideoSessionManager()
	_, cancel := context.WithCancel(context.Background())
	defer cancel()

	mockPeer := &MockPeer{}
	session1 := &videoPlaybackSession{
		cancel: cancel,
		done:   make(chan error, 1),
		peer:   mockPeer,
	}
	session2 := &videoPlaybackSession{
		cancel: cancel,
		done:   make(chan error, 1),
		peer:   mockPeer,
	}

	_ = manager.CreateSession("test-session", session1)

	// Act
	err := manager.CreateSession("test-session", session2)

	// Assert
	assert.Error(t, err)
	assert.ErrorIs(t, err, ErrSessionAlreadyExists)
}

func TestVideoSessionManager_GetSession(t *testing.T) {
	// Arrange
	manager := NewVideoSessionManager()
	_, cancel := context.WithCancel(context.Background())
	defer cancel()

	mockPeer := &MockPeer{}
	expectedSession := &videoPlaybackSession{
		cancel: cancel,
		done:   make(chan error, 1),
		peer:   mockPeer,
	}
	_ = manager.CreateSession("test-session", expectedSession)

	// Act
	session, exists := manager.GetSession("test-session")

	// Assert
	assert.True(t, exists)
	assert.Equal(t, expectedSession, session)
}

func TestVideoSessionManager_GetSession_NotFound(t *testing.T) {
	// Arrange
	manager := NewVideoSessionManager()

	// Act
	session, exists := manager.GetSession("non-existent")

	// Assert
	assert.False(t, exists)
	assert.Nil(t, session)
}

func TestVideoSessionManager_DeleteSession(t *testing.T) {
	// Arrange
	manager := NewVideoSessionManager()
	_, cancel := context.WithCancel(context.Background())
	defer cancel()

	mockPeer := &MockPeer{}
	session := &videoPlaybackSession{
		cancel: cancel,
		done:   make(chan error, 1),
		peer:   mockPeer,
	}
	_ = manager.CreateSession("test-session", session)

	// Act
	manager.DeleteSession("test-session")

	// Assert
	assert.False(t, manager.SessionExists("test-session"))
}

func TestVideoSessionManager_SessionExists(t *testing.T) {
	// Arrange
	manager := NewVideoSessionManager()
	_, cancel := context.WithCancel(context.Background())
	defer cancel()

	mockPeer := &MockPeer{}
	session := &videoPlaybackSession{
		cancel: cancel,
		done:   make(chan error, 1),
		peer:   mockPeer,
	}

	// Act & Assert
	assert.False(t, manager.SessionExists("test-session"))

	_ = manager.CreateSession("test-session", session)
	assert.True(t, manager.SessionExists("test-session"))
}
