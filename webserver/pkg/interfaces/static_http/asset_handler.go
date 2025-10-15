package static_http

import "net/http"

type AssetHandler interface {
	ServeAsset(w http.ResponseWriter, r *http.Request)
	GetScriptTags() []ScriptTag
	ShouldCacheAssets() bool
}

type ScriptTag struct {
	Type   string
	Src    string
	Module bool
}

