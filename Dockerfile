#################################################################################
#                          DOCKERFILE ARGUMENTS                                 #
#################################################################################
ARG BASE_REGISTRY=ghcr.io/josnelihurt-code/learning-cuda
ARG BASE_TAG=latest
ARG TARGETARCH=amd64
ARG GOLANG_VERSION=1.4.0

#################################################################################
#                    INTERMEDIATE IMAGES REFERENCES                              #
#################################################################################
# Only the Go-compiled artifact is needed by the runtime image. CPP/integration
# artifacts are consumed exclusively by test/integration/Dockerfile.tests.
#################################################################################

FROM ${BASE_REGISTRY}/intermediate:golang-built-${GOLANG_VERSION}-${TARGETARCH} AS golang-built-ref

#################################################################################
#                            RUNTIME STAGE (FINAL)                              #
#################################################################################
# Minimal runtime image without CUDA runtime (no compiler/build tools)
#################################################################################

FROM ${BASE_REGISTRY}/base:runtime-base-${BASE_TAG}-${TARGETARCH} AS runtime-base

FROM runtime-base AS runtime

WORKDIR /app

COPY --from=golang-built-ref /artifacts/bin/server /app/server

COPY data/static_images/ /app/data/static_images/
COPY data/videos/ /app/data/videos/
COPY data/lena.png data/lena_grayscale.png data/with_expected.png /app/data/
COPY config/config.yaml /app/config/config.yaml

COPY config/config.production.yaml /app/config/config.production.yaml

COPY src/go_api/VERSION /app/src/go_api/VERSION
COPY proto/VERSION /app/proto/VERSION

EXPOSE 8080 8443

CMD ["/app/server", "-config=/app/config/config.yaml"]
