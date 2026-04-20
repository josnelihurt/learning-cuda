package domain

import (
	"time"
)

type FrameCallback func(frameData []byte, frameNumber int, timestamp time.Duration) error
