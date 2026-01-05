ARG ALPINE_VERSION=3.20

FROM alpine:${ALPINE_VERSION} AS runtime-base

RUN apk add --no-cache \
    ca-certificates \
    curl \
    ffmpeg

WORKDIR /app

