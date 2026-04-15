package steps

type TestContext struct {
	*BDDContext
	serviceURL string
}

func NewTestContext() *TestContext {
	return &TestContext{}
}

func (tc *TestContext) Reset() {
	if tc.serviceURL != "" {
		tc.BDDContext = NewBDDContext(tc.serviceURL)
	}
}

func (tc *TestContext) CloseWebSocket() {}
