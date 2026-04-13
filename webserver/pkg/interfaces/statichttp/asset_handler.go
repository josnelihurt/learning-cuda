package statichttp

import "net/http"

type AssetHandler interface {
	ServeAsset(w http.ResponseWriter, r *http.Request)
	GetScriptTags(route string) []ScriptTag
	ShouldCacheAssets() bool
}

type ScriptTag struct {
	Type   string
	Src    string
	Module bool
}
