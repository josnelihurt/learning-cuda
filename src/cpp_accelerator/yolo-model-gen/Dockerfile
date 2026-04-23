ARG PYTHON_VERSION=3.12

# ---------- Stage 1: tools (pushed to GHCR) ----------
FROM python:${PYTHON_VERSION}-slim AS tools

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      ultralytics \
      onnx \
      onnxslim

WORKDIR /work

# ---------- Stage 2: generate (intermediate, not pushed) ----------
FROM tools AS generate

ARG YOLO_MODEL=yolov10n

COPY scripts/models/export_yolo_to_onnx.py /work/export_yolo_to_onnx.py

RUN python /work/export_yolo_to_onnx.py --model "${YOLO_MODEL}" --output /models/ \
    && ls -lh /models/

# ---------- Stage 3: artifact (pushed to GHCR, minimal) ----------
FROM scratch AS artifact

COPY --from=generate /models/ /models/
