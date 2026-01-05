package video

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os/exec"
	"time"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"github.com/rs/zerolog/log"
)

// FFmpegVideoPlayer decodes video files using ffmpeg and provides frames as domain.Image
type FFmpegVideoPlayer struct {
	videoPath string
	width     int
	height    int
	fps       float64
}

// NewFFmpegVideoPlayer creates a new video player for the given file
func NewFFmpegVideoPlayer(videoPath string) (*FFmpegVideoPlayer, error) {
	// Probe video to get dimensions and fps
	width, height, fps, err := probeVideo(videoPath)
	if err != nil {
		return nil, fmt.Errorf("failed to probe video: %w", err)
	}

	return &FFmpegVideoPlayer{
		videoPath: videoPath,
		width:     width,
		height:    height,
		fps:       fps,
	}, nil
}

// probeVideo uses ffprobe to get video metadata
func probeVideo(videoPath string) (width, height int, fps float64, err error) {
	// Get video dimensions
	cmd := exec.Command("ffprobe",
		"-v", "error",
		"-select_streams", "v:0",
		"-show_entries", "stream=width,height,r_frame_rate",
		"-of", "default=noprint_wrappers=1",
		videoPath,
	)

	output, err := cmd.CombinedOutput()
	if err != nil {
		return 0, 0, 0, fmt.Errorf("ffprobe failed: %w, output: %s", err, string(output))
	}

	// Parse output (format: width=1280\nheight=720\nr_frame_rate=25/1)
	var fpsNum, fpsDen int
	_, err = fmt.Sscanf(string(output), "width=%d\nheight=%d\nr_frame_rate=%d/%d",
		&width, &height, &fpsNum, &fpsDen)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("failed to parse ffprobe output: %w", err)
	}

	if fpsDen > 0 {
		fps = float64(fpsNum) / float64(fpsDen)
	} else {
		fps = 30.0 // Default to 30 fps
	}

	return width, height, fps, nil
}

// Play starts video playback and calls the frameCallback for each decoded frame
// The callback receives the raw RGB image data
func (p *FFmpegVideoPlayer) Play(ctx context.Context, frameCallback func(*domain.Image, int, time.Duration) error) error { //nolint:gocyclo // Complex due to video frame processing pipeline with context management
	log.Info().
		Str("video", p.videoPath).
		Int("width", p.width).
		Int("height", p.height).
		Float64("fps", p.fps).
		Msg("Starting video playback with FFmpeg")

	// Start ffmpeg to decode video to raw RGB frames
	// #nosec G204 -- videoPath is validated by caller from trusted video repository
	cmd := exec.CommandContext(ctx,
		"ffmpeg",
		"-i", p.videoPath,
		"-f", "rawvideo",
		"-pix_fmt", "rgb24",
		"-an",    // No audio
		"-sn",    // No subtitles
		"pipe:1", // Output to stdout
	)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdout pipe: %w", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("failed to create stderr pipe: %w", err)
	}

	// Start ffmpeg process
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start ffmpeg: %w", err)
	}

	// Log ffmpeg stderr in background
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			log.Debug().Str("ffmpeg", scanner.Text()).Msg("FFmpeg output")
		}
	}()

	// Calculate frame size (RGB = 3 bytes per pixel)
	frameSize := p.width * p.height * 3
	frameBuffer := make([]byte, frameSize)

	frameNumber := 0
	frameDuration := time.Duration(float64(time.Second) / p.fps)
	startTime := time.Now()

	reader := bufio.NewReader(stdout)

	for {
		select {
		case <-ctx.Done():
			log.Info().Msg("Video playback stopped by context")
			if cmd.Process != nil {
				// Fixed: Check kill error to detect zombie processes
				// If kill fails, log the error but don't fail the context cancellation
				if err := cmd.Process.Kill(); err != nil && !isProcessFinished(err) {
					log.Warn().Err(err).Msg("Failed to kill ffmpeg process, may be zombie")
					// Note: Process may have already exited, which is acceptable
				}
			}
			// Clean up pipes
			_ = stdout.Close()
			_ = stderr.Close()
			return ctx.Err()
		default:
			// Read one complete frame
			n, err := io.ReadFull(reader, frameBuffer)
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				log.Info().Int("frames", frameNumber).Msg("Video playback completed")
				return nil
			}
			if err != nil {
				log.Error().Err(err).Msg("Error reading frame from ffmpeg")
				return fmt.Errorf("failed to read frame: %w", err)
			}
			if n != frameSize {
				return fmt.Errorf("incomplete frame read: got %d bytes, expected %d", n, frameSize)
			}

			// Create domain image
			img := &domain.Image{
				Data:   make([]byte, frameSize),
				Width:  p.width,
				Height: p.height,
			}
			copy(img.Data, frameBuffer)

			// Calculate timestamp
			timestamp := time.Duration(frameNumber) * frameDuration

			// Call the frame callback
			if err := frameCallback(img, frameNumber, timestamp); err != nil {
				log.Error().Err(err).Int("frame", frameNumber).Msg("Frame callback error")
				return err
			}

			frameNumber++

			// Log progress every 30 frames (~1 second)
			if frameNumber%30 == 0 {
				elapsed := time.Since(startTime)
				actualFPS := float64(frameNumber) / elapsed.Seconds()
				log.Debug().
					Int("frame", frameNumber).
					Float64("fps", actualFPS).
					Dur("elapsed", elapsed).
					Msg("Video playback progress")
			}
		}
	}
}

// GetDimensions returns the video dimensions
func (p *FFmpegVideoPlayer) GetDimensions() (width, height int) {
	return p.width, p.height
}

// GetFPS returns the video frame rate
func (p *FFmpegVideoPlayer) GetFPS() float64 {
	return p.fps
}

// isProcessFinished checks if an error from Process.Kill indicates the process was already finished
func isProcessFinished(err error) bool {
	// Process.Signal and Process.Kill return an error if the process has already exited.
	// On Unix systems, this is typically os.ErrProcessDone.
	// On Windows, the error message may contain "already finished".
	// For now, we check common patterns indicating the process was done.
	return err != nil && (err.Error() == "os: process already finished" ||
		err.Error() == "wait: no child processes" ||
		containsString(err.Error(), "already finished"))
}

// containsString is a simple helper for substring check
func containsString(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr ||
		len(s) > len(substr) && contains(s, substr))
}

// contains checks if substr is in s
func contains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
