package config

type LoggerConfig struct {
	Level         string `mapstructure:"level"`
	Format        string `mapstructure:"format"`
	Output        string `mapstructure:"output"`
	FilePath      string `mapstructure:"file_path"`
	IncludeCaller bool   `mapstructure:"include_caller"`
}
