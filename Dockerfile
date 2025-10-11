FROM node:20-alpine AS frontend-builder

WORKDIR /build

COPY webserver/web/package*.json ./
RUN npm ci

COPY webserver/web/ ./
RUN npm run build

FROM nvidia/cuda:12.5.1-devel-ubuntu24.04 AS backend-builder

ENV DEBIAN_FRONTEND=noninteractive
ENV BAZEL_VERSION=7.0.2
ENV GO_VERSION=1.23.0

WORKDIR /build

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    pkg-config \
    zip \
    unzip \
    python3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64 \
    && chmod +x /usr/local/bin/bazel

RUN wget https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz \
    && rm go${GO_VERSION}.linux-amd64.tar.gz

ENV PATH="/usr/local/go/bin:${PATH}"
ENV GOPATH="/go"
ENV GO111MODULE=on

COPY . .

RUN bazel build \
    //webserver/cmd/server:server \
    //cpp_accelerator/ports/cgo:cgo_api

RUN mkdir -p /artifacts/bin /artifacts/lib && \
    cp -L bazel-bin/webserver/cmd/server/server_/server /artifacts/bin/ && \
    find bazel-bin/cpp_accelerator/ports/cgo -name "*.so" -exec cp -L {} /artifacts/lib/ \; || true && \
    find bazel-bin/cpp_accelerator/ports/cgo -name "libcgo_api.a" -exec cp -L {} /artifacts/lib/ \; || true

FROM nvidia/cuda:12.5.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=backend-builder /artifacts/bin/server /app/server
COPY --from=backend-builder /artifacts/lib/ /usr/local/lib/
COPY --from=frontend-builder /build/static/ /app/web/static/
COPY --from=frontend-builder /build/templates/ /app/web/templates/
COPY --from=backend-builder /build/data/ /app/data/

RUN ldconfig

EXPOSE 8080

ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

CMD ["/app/server", "-webroot=/app/web"]
