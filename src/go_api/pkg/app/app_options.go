package app

import "connectrpc.com/connect"

type Option func(*App)

func WithInterceptors(interceptors ...connect.Interceptor) Option {
	return func(a *App) {
		a.interceptors = append(a.interceptors, interceptors...)
	}
}
