package connectrpc

import (
	"context"
	"net/http"
)

type httpClient interface {
	Do(req *http.Request) (*http.Response, error)
}

type featureFlagsManager interface {
	Iterate(ctx context.Context, fn func(ctx context.Context, flagKey string, flagValue interface{}) error) error
}
