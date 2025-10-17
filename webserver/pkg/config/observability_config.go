package config

type ObservabilityConfig struct {
	Enabled               bool    `mapstructure:"enabled"`
	ServiceName           string  `mapstructure:"service_name"`
	ServiceVersion        string  `mapstructure:"service_version"`
	OtelCollectorEndpoint string  `mapstructure:"otel_collector_endpoint"`
	TraceSamplingRate     float64 `mapstructure:"trace_sampling_rate"`
}
