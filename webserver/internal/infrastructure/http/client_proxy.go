package http

import (
	"context"
	"io"
	"net/http"
	"net/url"
	"strings"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
)

type clientProxy struct {
	client *http.Client
}

func New(client *http.Client) *clientProxy {
	client.Transport = otelhttp.NewTransport(client.Transport)
	return &clientProxy{client: client}
}

func (p *clientProxy) Do(req *http.Request) (*http.Response, error) {
	tracer := otel.Tracer("httpinfra.ClientProxy")
	ctx, span := tracer.Start(req.Context(), "HTTP.Do:" + req.Method + "=" + req.URL.Path)
	defer span.End()

	span.SetAttributes(
		attribute.String("http.client", "http.client_proxy"),
		attribute.String("http.method", req.Method),
		attribute.String("http.url", req.URL.String()),
		attribute.String("http.host", req.URL.Host),
		attribute.String("http.path", req.URL.Path),
	)

	req = req.WithContext(ctx)
	resp, err := p.client.Do(req)
	
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, "HTTP request failed")
		return nil, err
	}

	span.SetAttributes(
		attribute.Int("http.status_code", resp.StatusCode),
	)

	if resp.StatusCode >= 400 {
		span.SetStatus(codes.Error, "HTTP error status code")
	} else {
		span.SetStatus(codes.Ok, "HTTP request successful")
	}

	return resp, nil
}

func (p *clientProxy) Get(url string) (*http.Response, error) {
	tracer := otel.Tracer("httpinfra.ClientProxy")
	ctx, span := tracer.Start(context.Background(), "HTTP.Get")
	defer span.End()

	span.SetAttributes(
		attribute.String("http.method", http.MethodGet),
		attribute.String("http.url", url),
	)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, "Failed to create request")
		return nil, err
	}
	
	resp, doErr := p.Do(req)
	if doErr != nil {
		span.RecordError(doErr)
		span.SetStatus(codes.Error, "Request failed")
		return nil, doErr
	}
	
	span.SetStatus(codes.Ok, "Request successful")
	return resp, nil
}

func (p *clientProxy) GetWithContext(ctx context.Context, url string) (*http.Response, error) {
	tracer := otel.Tracer("httpinfra.ClientProxy")
	ctx, span := tracer.Start(ctx, "HTTP.Get")
	defer span.End()

	span.SetAttributes(
		attribute.String("http.method", http.MethodGet),
		attribute.String("http.url", url),
	)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, "Failed to create request")
		return nil, err
	}
	
	resp, doErr := p.Do(req)
	if doErr != nil {
		span.RecordError(doErr)
		span.SetStatus(codes.Error, "Request failed")
		return nil, doErr
	}
	
	span.SetStatus(codes.Ok, "Request successful")
	return resp, nil
}

func (p *clientProxy) Head(url string) (*http.Response, error) {
	tracer := otel.Tracer("httpinfra.ClientProxy")
	ctx, span := tracer.Start(context.Background(), "HTTP.Head")
	defer span.End()

	span.SetAttributes(
		attribute.String("http.method", http.MethodHead),
		attribute.String("http.url", url),
	)

	req, err := http.NewRequestWithContext(ctx, http.MethodHead, url, nil)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, "Failed to create request")
		return nil, err
	}
	
	resp, doErr := p.Do(req)
	if doErr != nil {
		span.RecordError(doErr)
		span.SetStatus(codes.Error, "Request failed")
		return nil, doErr
	}
	
	span.SetStatus(codes.Ok, "Request successful")
	return resp, nil
}

func (p *clientProxy) HeadWithContext(ctx context.Context, url string) (*http.Response, error) {
	tracer := otel.Tracer("httpinfra.ClientProxy")
	ctx, span := tracer.Start(ctx, "HTTP.Head")
	defer span.End()

	span.SetAttributes(
		attribute.String("http.method", http.MethodHead),
		attribute.String("http.url", url),
	)

	req, err := http.NewRequestWithContext(ctx, http.MethodHead, url, nil)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, "Failed to create request")
		return nil, err
	}
	
	resp, doErr := p.Do(req)
	if doErr != nil {
		span.RecordError(doErr)
		span.SetStatus(codes.Error, "Request failed")
		return nil, doErr
	}
	
	span.SetStatus(codes.Ok, "Request successful")
	return resp, nil
}

func (p *clientProxy) Post(url, contentType string, body io.Reader) (*http.Response, error) {
	tracer := otel.Tracer("httpinfra.ClientProxy")
	ctx, span := tracer.Start(context.Background(), "HTTP.Post")
	defer span.End()

	span.SetAttributes(
		attribute.String("http.method", http.MethodPost),
		attribute.String("http.url", url),
		attribute.String("http.content_type", contentType),
	)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, body)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, "Failed to create request")
		return nil, err
	}
	req.Header.Set("Content-Type", contentType)
	
	resp, doErr := p.Do(req)
	if doErr != nil {
		span.RecordError(doErr)
		span.SetStatus(codes.Error, "Request failed")
		return nil, doErr
	}
	
	span.SetStatus(codes.Ok, "Request successful")
	return resp, nil
}

func (p *clientProxy) PostWithContext(ctx context.Context, url, contentType string, body io.Reader) (*http.Response, error) {
	tracer := otel.Tracer("httpinfra.ClientProxy")
	ctx, span := tracer.Start(ctx, "HTTP.Post")
	defer span.End()

	span.SetAttributes(
		attribute.String("http.method", http.MethodPost),
		attribute.String("http.url", url),
		attribute.String("http.content_type", contentType),
	)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, body)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, "Failed to create request")
		return nil, err
	}
	req.Header.Set("Content-Type", contentType)
	
	resp, doErr := p.Do(req)
	if doErr != nil {
		span.RecordError(doErr)
		span.SetStatus(codes.Error, "Request failed")
		return nil, doErr
	}
	
	span.SetStatus(codes.Ok, "Request successful")
	return resp, nil
}

func (p *clientProxy) PostForm(url string, data url.Values) (*http.Response, error) {
	tracer := otel.Tracer("httpinfra.ClientProxy")
	ctx, span := tracer.Start(context.Background(), "HTTP.PostForm")
	defer span.End()

	span.SetAttributes(
		attribute.String("http.method", http.MethodPost),
		attribute.String("http.url", url),
		attribute.String("http.content_type", "application/x-www-form-urlencoded"),
		attribute.Int("http.form_fields", len(data)),
	)

	resp, err := p.PostWithContext(ctx, url, "application/x-www-form-urlencoded", strings.NewReader(data.Encode()))
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, "PostForm failed")
		return nil, err
	}
	
	span.SetStatus(codes.Ok, "PostForm successful")
	return resp, nil
}

func (p *clientProxy) PostFormWithContext(ctx context.Context, url string, data url.Values) (*http.Response, error) {
	tracer := otel.Tracer("httpinfra.ClientProxy")
	ctx, span := tracer.Start(ctx, "HTTP.PostForm")
	defer span.End()

	span.SetAttributes(
		attribute.String("http.method", http.MethodPost),
		attribute.String("http.url", url),
		attribute.String("http.content_type", "application/x-www-form-urlencoded"),
		attribute.Int("http.form_fields", len(data)),
	)

	resp, err := p.PostWithContext(ctx, url, "application/x-www-form-urlencoded", strings.NewReader(data.Encode()))
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, "PostForm failed")
		return nil, err
	}
	
	span.SetStatus(codes.Ok, "PostForm successful")
	return resp, nil
}

func (p *clientProxy) CloseIdleConnections() {
	p.client.CloseIdleConnections()
}

