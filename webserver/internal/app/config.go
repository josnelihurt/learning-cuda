package app

import "flag"

type Config struct {
	Port        string
	DevMode     bool
	WebRootPath string
}

func LoadConfig() *Config {
	cfg := &Config{}
	flag.StringVar(&cfg.Port, "port", ":8080", "Server port")
	flag.BoolVar(&cfg.DevMode, "dev", false, "Enable development mode")
	flag.StringVar(&cfg.WebRootPath, "webroot", "webserver/web", "Path to web assets")
	flag.Parse()
	return cfg
}

