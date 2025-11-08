ARG GO_VERSION=1.24
ARG ALPINE_VERSION=3.19

FROM golang:${GO_VERSION}-alpine AS proto-builder

ARG BUF_VERSION=v1.47.2
ARG PROTOC_GEN_CONNECT_GO_VERSION=v1.19.1
ARG PROTOC_GEN_GO_VERSION=v1.35.2

RUN go install github.com/bufbuild/buf/cmd/buf@${BUF_VERSION} && \
    go install connectrpc.com/connect/cmd/protoc-gen-connect-go@${PROTOC_GEN_CONNECT_GO_VERSION} && \
    go install google.golang.org/protobuf/cmd/protoc-gen-go@${PROTOC_GEN_GO_VERSION}

FROM alpine:${ALPINE_VERSION}
RUN apk add --update --no-cache \
    ca-certificates \
    git \
    protoc \
    openssh-client \
    nodejs \
    npm && \
  rm -rf /var/cache/apk/*

COPY --from=proto-builder /go/bin/buf /usr/local/bin/buf
COPY --from=proto-builder /go/bin/protoc-gen-go /usr/local/bin/protoc-gen-go
COPY --from=proto-builder /go/bin/protoc-gen-connect-go /usr/local/bin/protoc-gen-connect-go

WORKDIR /workspace
ENV XDG_CACHE_HOME=/workspace/.cache

ENTRYPOINT ["/usr/local/bin/buf"]

