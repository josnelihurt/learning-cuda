package video

import "context"

type videoPlaybackSession struct {
	cancel context.CancelFunc
	done   chan error
	peer   StreamVideoPeer
}
