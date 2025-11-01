FROM golang:1.24-alpine AS proto-builder
RUN go install github.com/bufbuild/buf/cmd/buf@v1.47.2
RUN go install connectrpc.com/connect/cmd/protoc-gen-connect-go@v1.19.1
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.35.2

FROM alpine:3.19
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

