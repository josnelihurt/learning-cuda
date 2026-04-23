#!/usr/bin/env bash
# Fix CUDA/TRT version mismatch on the dev PC (x86, driver 575).
#
# Root cause: libnvinfer was installed from the +cuda13.2 channel, but driver
# 575.57.08 only supports up to CUDA 12.9.  TRT's createInferBuilder therefore
# fails with "CUDA initialization failure with error: 35".
#
# This script:
#   1. Installs the CUDA 12.9 toolkit (cudart, headers, etc.)
#   2. Switches every TRT package to the +cuda12.9 variant
#   3. Rebuilds the C++ accelerator binary
#
# Run once as a user with sudo access:
#   sudo bash scripts/dev/fix-cuda-trt-versions.sh
# -------------------------------------------------------------------
set -euo pipefail

TRT_VERSION="10.16.1.11-1+cuda12.9"

echo "==> Installing CUDA 12.9 toolkit..."
apt-get install -y \
  cuda-toolkit-12-9 \
  cuda-cudart-12-9

echo "==> Switching TensorRT packages to +cuda12.9..."
apt-get install -y --allow-downgrades \
  libnvinfer10="${TRT_VERSION}" \
  libnvinfer-dev="${TRT_VERSION}" \
  libnvinfer-headers-dev="${TRT_VERSION}" \
  libnvinfer-headers-plugin-dev="${TRT_VERSION}" \
  libnvinfer-safe-headers-dev="${TRT_VERSION}" \
  libnvinfer-plugin10="${TRT_VERSION}" \
  libnvinfer-plugin-dev="${TRT_VERSION}" \
  libnvinfer-bin="${TRT_VERSION}" \
  libnvinfer-dispatch10="${TRT_VERSION}" \
  libnvinfer-lean10="${TRT_VERSION}" \
  libnvinfer-vc-plugin10="${TRT_VERSION}" \
  libnvonnxparsers10="${TRT_VERSION}" \
  libnvonnxparsers-dev="${TRT_VERSION}"

echo "==> Removing old CUDA 12.5 packages..."
# Remove the CUDA 12.5 meta-packages; autoremove cleans up their dependencies
apt-get remove --purge -y \
  cuda-toolkit-12-5 \
  cuda-command-line-tools-12-5 \
  cuda-libraries-12-5 \
  cuda-libraries-dev-12-5 \
  cuda-tools-12-5 \
  cuda-visual-tools-12-5 \
  gds-tools-12-5 || true
apt-get autoremove --purge -y

echo "==> Done. Now rebuild:"
echo "    bazel build //src/cpp_accelerator/ports/grpc:accelerator_control_client"
