package config

type ProcessorConfig struct {
	LibraryBasePath string       `mapstructure:"library_base_path"`
	DefaultLibrary  string       `mapstructure:"default_library"`
	ListenAddress   string       `mapstructure:"listen_address"`  // control-stream server address
	TLS             ProcessorTLS `mapstructure:"tls"`
}

type ProcessorTLS struct {
	CertFile     string `mapstructure:"cert_file"`      // server cert presented to accelerators
	KeyFile      string `mapstructure:"key_file"`       // server private key
	ClientCAFile string `mapstructure:"client_ca_file"` // CA bundle that signed accelerator client certs (mTLS)
}
