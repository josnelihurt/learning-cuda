package domain

type FeatureFlag struct {
	Key          string
	Name         string
	Type         FeatureFlagType
	Enabled      bool
	DefaultValue interface{}
}

type FeatureFlagType string

const (
	BooleanFlagType FeatureFlagType = "boolean"
	VariantFlagType FeatureFlagType = "variant"
)

type FeatureFlagEvaluation struct {
	FlagKey      string
	EntityID     string
	Result       interface{}
	Success      bool
	UsedFallback bool
}
