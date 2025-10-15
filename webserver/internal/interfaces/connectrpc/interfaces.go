package connectrpc

import (
	"context"
	"net/http"
)

type httpClient interface {
	Do(req *http.Request) (*http.Response, error)
}

type featureFlagsManager interface {
	Sync(ctx context.Context) error
}
