package config

type ProcessorConfig struct {
	LibraryBasePath string
	DefaultLibrary  string
	EnableHotReload bool
	FallbackEnabled bool
	FallbackChain   []string
}
