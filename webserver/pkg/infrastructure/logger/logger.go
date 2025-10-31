package logger

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/rs/zerolog"
	"go.opentelemetry.io/otel/trace"
)

var globalLogger zerolog.Logger

type Config struct {
	Level         string
	Format        string
	Output        string
	FilePath      string
	IncludeCaller bool
}

func New(cfg Config) zerolog.Logger {
	var output io.Writer = os.Stdout

	level, err := zerolog.ParseLevel(cfg.Level)
	if err != nil {
		level = zerolog.InfoLevel
	}

	zerolog.TimeFieldFormat = time.RFC3339Nano

	if cfg.Format == "console" {
		output = zerolog.ConsoleWriter{
			Out:        os.Stdout,
			TimeFormat: time.RFC3339,
		}
	}

	if cfg.Output == "file" && cfg.FilePath != "" {
		// Open file in append mode, create if not exists
		file, err := os.OpenFile(cfg.FilePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if err == nil {
			// Write to both stdout and file to ensure visibility and file shipping
			output = io.MultiWriter(os.Stdout, file)
		}
	}

	logger := zerolog.New(output).
		Level(level).
		With().
		Timestamp()

	if cfg.IncludeCaller {
		// Set up custom caller formatter to show only filename
		zerolog.CallerFieldName = "caller"
		zerolog.CallerMarshalFunc = func(pc uintptr, file string, line int) string {
			return filepath.Base(fmt.Sprintf("%s:%d", file, line))
		}
		logger = logger.CallerWithSkipFrameCount(1)
	}

	globalLogger = logger.Logger()
	return globalLogger
}

func Global() *zerolog.Logger {
	return &globalLogger
}

func FromContext(ctx context.Context) *zerolog.Logger {
	spanContext := trace.SpanContextFromContext(ctx)

	if !spanContext.IsValid() {
		return &globalLogger
	}

	logger := globalLogger.With().
		Str("trace_id", spanContext.TraceID().String()).
		Str("span_id", spanContext.SpanID().String()).
		Logger()

	return &logger
}

func WithTraceContext(ctx context.Context, logger *zerolog.Logger) *zerolog.Logger {
	spanContext := trace.SpanContextFromContext(ctx)

	if !spanContext.IsValid() {
		return logger
	}

	contextLogger := logger.With().
		Str("trace_id", spanContext.TraceID().String()).
		Str("span_id", spanContext.SpanID().String()).
		Logger()

	return &contextLogger
}
