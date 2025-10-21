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
// The callback receives the raw RGBA image data
func (p *FFmpegVideoPlayer) Play(ctx context.Context, frameCallback func(*domain.Image, int, time.Duration) error) error {
	log.Info().
		Str("video", p.videoPath).
		Int("width", p.width).
		Int("height", p.height).
		Float64("fps", p.fps).
		Msg("Starting video playback with FFmpeg")

	// Start ffmpeg to decode video to raw RGBA frames
	// #nosec G204 -- videoPath is validated by caller from trusted video repository
	cmd := exec.CommandContext(ctx,
		"ffmpeg",
		"-i", p.videoPath,
		"-f", "rawvideo",
		"-pix_fmt", "rgba",
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

	// Calculate frame size (RGBA = 4 bytes per pixel)
	frameSize := p.width * p.height * 4
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
				_ = cmd.Process.Kill() //nolint:errcheck // Best effort cleanup on context cancellation
			}
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
