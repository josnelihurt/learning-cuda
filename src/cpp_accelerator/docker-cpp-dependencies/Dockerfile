ARG BASE_REGISTRY=ghcr.io/josnelihurt-code/learning-cuda
ARG BASE_TAG=latest
ARG TARGETARCH=amd64

FROM ${BASE_REGISTRY}/base:bazel-base-${BASE_TAG}-${TARGETARCH}

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    && rm -rf /var/lib/apt/lists/*

COPY MODULE.bazel MODULE.bazel.lock WORKSPACE.bazel BUILD.bazel ./
COPY .bazelrc .bazeliskrc ./
COPY bazel/ ./bazel/
COPY proto/ ./proto/
COPY third_party/ ./third_party/
COPY buf.yaml buf.lock buf.gen.yaml ./

ENV USE_BAZEL_VERSION=8.4.2
RUN bazel --batch fetch //...

RUN bazel --batch build \
    --show_timestamps \
    --announce_rc \
    @spdlog//:spdlog \
    @com_google_protobuf//:protobuf \
    //proto:common_cc_proto \
    //proto:image_processor_service_cc_proto \
    //proto:config_service_cc_proto \
    //third_party/stb:stb_image \
    //third_party/stb:stb_image_write
