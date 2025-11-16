ARG CUDA_VERSION=12.5.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update --allow-releaseinfo-change || apt-get update || true && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/cuda/grpc

COPY image_processor_grpc_server ./bin/image_processor_grpc_server
COPY libcuda_processor.so ./lib/libcuda_processor.so

ENV LD_LIBRARY_PATH=/opt/cuda/grpc/lib:${LD_LIBRARY_PATH}

EXPOSE 60061

ENTRYPOINT ["./bin/image_processor_grpc_server"]

