package config

type ProcessorConfig struct {
	LibraryBasePath   string `mapstructure:"library_base_path"`
	GRPCServerAddress string `mapstructure:"grpc_server_address"`
	DefaultLibrary    string `mapstructure:"default_library"`
}
