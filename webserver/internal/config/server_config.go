package config

type ServerConfig struct {
	HTTPPort         string
	HTTPSPort        string
	HotReloadEnabled bool
	WebRootPath      string
	DevServerURL     string
	DevServerPaths   []string
	TLSConfig        TLSConfig
}