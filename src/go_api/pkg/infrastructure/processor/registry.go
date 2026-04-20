package processor

import (
	"errors"
	"sync"

	"github.com/rs/zerolog"
)

// Registry holds the set of registered accelerator sessions.
// v1 enforces exactly one session at a time; the map shape supports v2 multi-device.
type Registry struct {
	mu       sync.RWMutex
	sessions map[string]*AcceleratorSession
	log      zerolog.Logger
}

func NewRegistry(log zerolog.Logger) *Registry {
	return &Registry{
		sessions: make(map[string]*AcceleratorSession),
		log:      log,
	}
}

// Add registers a new session. Returns an error if a session with the same
// device_id is already registered (v1 single-accelerator policy).
func (r *Registry) Add(s *AcceleratorSession) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.sessions) > 0 {
		return errors.New("v1 supports only one accelerator at a time")
	}
	r.sessions[s.DeviceID] = s
	r.log.Info().Str("device_id", s.DeviceID).Msg("accelerator session registered")
	return nil
}

// Remove removes the session for the given device_id.
func (r *Registry) Remove(deviceID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.sessions, deviceID)
	r.log.Info().Str("device_id", deviceID).Msg("accelerator session removed")
}

// Get returns the session for the given device_id, or nil + false.
func (r *Registry) Get(deviceID string) (*AcceleratorSession, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	s, ok := r.sessions[deviceID]
	return s, ok
}

// First returns the singleton session in v1, or nil + false if none registered.
// v2 will deprecate this in favour of explicit device selection.
func (r *Registry) First() (*AcceleratorSession, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, s := range r.sessions {
		return s, true
	}
	return nil, false
}
