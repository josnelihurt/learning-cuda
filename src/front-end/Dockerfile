ARG BASE_REGISTRY=ghcr.io/josnelihurt/learning-cuda
ARG BASE_TAG=latest
ARG TARGETARCH=amd64

FROM ${BASE_REGISTRY}/base:proto-tools-${BASE_TAG}-${TARGETARCH} AS proto-tools

FROM node:20-alpine AS builder

WORKDIR /build

RUN apk add --no-cache protobuf-dev

COPY --from=proto-tools /go/bin/buf /usr/local/bin/buf
COPY --from=proto-tools /go/bin/protoc-gen-go /usr/local/bin/protoc-gen-go
COPY --from=proto-tools /go/bin/protoc-gen-connect-go /usr/local/bin/protoc-gen-connect-go
COPY --from=proto-tools /go/bin/protoc-gen-go-grpc /usr/local/bin/protoc-gen-go-grpc

COPY buf.yaml buf.lock buf.gen.yaml ./
COPY proto ./proto

COPY src/front-end/package*.json ./src/front-end/
RUN cd src/front-end && npm ci

COPY src/front-end/ ./src/front-end/

RUN buf generate

WORKDIR /build/src/front-end
RUN npm run build

FROM nginx:1.27-alpine

COPY --from=builder /build/src/front-end/dist /usr/share/nginx/html
COPY src/front-end/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
