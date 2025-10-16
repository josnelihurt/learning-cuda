package config

type LoggerConfig struct {
	Level         string
	Format        string
	Output        string
	IncludeCaller bool
}
