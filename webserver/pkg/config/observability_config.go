package config

type ObservabilityConfig struct {
	Enabled                   bool    `mapstructure:"enabled"`
	ServiceName               string  `mapstructure:"service_name"`
	ServiceVersion            string  `mapstructure:"service_version"`
	OtelCollectorGRPCEndpoint string  `mapstructure:"otel_collector_grpc_endpoint"`
	OtelCollectorHTTPEndpoint string  `mapstructure:"otel_collector_http_endpoint"`
	TraceSamplingRate         float64 `mapstructure:"trace_sampling_rate"`
}
