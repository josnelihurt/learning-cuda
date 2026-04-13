package steps

type TestContext struct {
	*BDDContext
	fliptURL   string
	namespace  string
	serviceURL string
}

func NewTestContext() *TestContext {
	return &TestContext{}
}

func (tc *TestContext) Reset() {
	if tc.BDDContext != nil {
		tc.BDDContext.CloseWebSocket()
	}
	if tc.fliptURL != "" && tc.serviceURL != "" {
		tc.BDDContext = NewBDDContext(
			tc.fliptURL,
			tc.namespace,
			tc.serviceURL,
		)
	}
}

func (tc *TestContext) CloseWebSocket() {
	if tc.BDDContext != nil {
		tc.BDDContext.CloseWebSocket()
	}
}
