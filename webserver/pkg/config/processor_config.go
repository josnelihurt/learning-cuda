package config

type ProcessorConfig struct {
	LibraryBasePath string   `mapstructure:"library_base_path"`
	DefaultLibrary  string   `mapstructure:"default_library"`
	EnableHotReload bool     `mapstructure:"enable_hot_reload"`
	FallbackEnabled bool     `mapstructure:"fallback_enabled"`
	FallbackChain   []string `mapstructure:"fallback_chain"`
}
