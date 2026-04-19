package config

type ProcessorConfig struct {
	LibraryBasePath   string       `mapstructure:"library_base_path"`
	GRPCServerAddress string       `mapstructure:"grpc_server_address"` // legacy forward-dial; removed in step-09
	DefaultLibrary    string       `mapstructure:"default_library"`
	ListenAddress     string       `mapstructure:"listen_address"`  // control-stream server address
	TLS               ProcessorTLS `mapstructure:"tls"`
}

type ProcessorTLS struct {
	CertFile     string `mapstructure:"cert_file"`      // server cert presented to accelerators
	KeyFile      string `mapstructure:"key_file"`       // server private key
	ClientCAFile string `mapstructure:"client_ca_file"` // CA bundle that signed accelerator client certs (mTLS)
}
