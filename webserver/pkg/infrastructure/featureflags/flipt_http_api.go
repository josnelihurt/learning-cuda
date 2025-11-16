package featureflags

// FliptHTTPAPI is a thin wrapper around FliptWriter that is kept for
// backwards compatibility with existing code and tests.
type FliptHTTPAPI struct {
	*FliptWriter
}

// NewFliptHTTPAPI constructs a FliptHTTPAPI backed by a FliptWriter.
func NewFliptHTTPAPI(grpcURL, namespace string, client httpClient) *FliptHTTPAPI {
	return &FliptHTTPAPI{
		FliptWriter: NewFliptWriter(grpcURL, namespace, client),
	}
}
