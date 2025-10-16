package logger

import (
	"context"
	"io"
	"os"
	"time"

	"github.com/rs/zerolog"
	"go.opentelemetry.io/otel/trace"
)

var globalLogger zerolog.Logger

type Config struct {
	Level         string
	Format        string
	Output        string
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

	logger := zerolog.New(output).
		Level(level).
		With().
		Timestamp()

	if cfg.IncludeCaller {
		logger = logger.Caller()
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
