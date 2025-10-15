package config

type ObservabilityConfig struct {
	enabled               bool
	ServiceName           string
	ServiceVersion        string
	OtelCollectorEndpoint string
	TraceSamplingRate     float64
}
