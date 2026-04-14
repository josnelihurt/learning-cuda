package domain

type FeatureFlag struct {
	Key          string
	Name         string
	Type         FeatureFlagType
	Enabled      bool
	DefaultValue interface{}
	Description  string
}

type FeatureFlagType string

const (
	BooleanFlagType FeatureFlagType = "boolean"
	StringFlagType  FeatureFlagType = "string"
)

type FeatureFlagEvaluation struct {
	FlagKey      string
	EntityID     string
	Result       interface{}
	Success      bool
	UsedFallback bool
}
