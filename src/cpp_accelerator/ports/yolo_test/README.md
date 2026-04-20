# YOLO Integration Test

This directory contains the validation example for YOLO object detection integration into the CUDA Learning Platform.

## Prerequisites

### System Requirements
- CUDA 12.5+ installed
- cuDNN 9 for CUDA 12
- NVIDIA GPU with compute capability compatible with CUDA 12.5
- Bazel 8.4.2+
- C++23 compiler

### Installation Steps (Ubuntu 24.04)

#### 1. Install CUDA 12.5
```bash
# Follow NVIDIA's official guide or use:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-12-5
```

#### 2. Install cuDNN 9 for CUDA 12
```bash
# Add NVIDIA repository and install cuDNN
sudo apt-get update
sudo apt-get -y install cudnn9-cuda-12
```

#### 3. Install ONNX Runtime
```bash
# Download and extract ONNX Runtime v1.19.2
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-gpu-1.19.2.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.19.2.tgz

# Install libraries
cd onnxruntime-linux-x64-gpu-1.19.2/lib/
sudo cp libonnxruntime.so* /usr/local/lib/
sudo cp libonnxruntime_providers_shared.so /usr/local/lib/
sudo cp libonnxruntime_providers_cuda.so /usr/local/lib/
sudo cp libonnxruntime_providers_tensorrt.so /usr/local/lib/
sudo ldconfig

# Install headers
cd ../include
sudo cp -r * /usr/local/include/
```

#### 4. Verify Installation
```bash
# Check CUDA
nvidia-smi
nvcc --version

# Check cuDNN
ldconfig -p | grep cudnn

# Check ONNX Runtime
ls /usr/local/lib/libonnxruntime.so
```

## Building the Test

```bash
# From the project root
bazel build //src/cpp_accelerator/ports/yolo_test:yolo_validation_example
```

## Running the Test

### 1. Export YOLO Model to ONNX
```bash
# Install ultralytics
pip install ultralytics

# Export YOLOv10n model
python scripts/models/export_yolo_to_onnx.py --model yolov10n
```

This creates `data/models/yolov10n.onnx` (~9.9 MB).

### 2. Run Validation
```bash
./bazel-bin/src/cpp_accelerator/ports/yolo_test/yolo_validation_example
```

### Expected Output
```
[INFO] YOLO Validation Example - Starting...
[INFO] Loading YOLO model from: data/models/yolov10n.onnx
[INFO] YOLODetector initialized with model: data/models/yolov10n.onnx
[INFO] YOLO model loaded successfully!
[INFO] CUDA Execution Provider: Active
[INFO] Test image loaded: 512x512
[INFO] Running object detection...
[INFO] YOLO detected X objects...
[INFO] ✓ YOLO validation PASSED - Library initialized successfully!
```

## Architecture

### YOLO Detector Integration
- **Location**: `src/cpp_accelerator/infrastructure/cuda/yolo_detector.{h,cpp}`
- **Pattern**: Follows existing IFilter pattern
- **Dependencies**: ONNX Runtime C++ API with CUDA Execution Provider

### Data Flow
```
ImageBuffer → YOLODetector::Apply()
    ↓
Preprocess (resize to 640x640, normalize)
    ↓
ONNX Runtime Inference (CUDA)
    ↓
Postprocess (NMS, confidence threshold)
    ↓
std::vector<Detection>
```

## Troubleshooting

### Error: `libcudnn.so.9: cannot open shared object file`
**Solution**: Install cuDNN 9 for CUDA 12:
```bash
sudo apt-get -y install cudnn9-cuda-12
```

### Error: `libonnxruntime.so.1: cannot open shared object file`
**Solution**: Install ONNX Runtime (see Installation Steps above).

### Error: `CUDNN_STATUS_NOT_INITIALIZED`
**Solution**: Verify cuDNN version matches CUDA version:
```bash
# You need cuDNN 9 for CUDA 12, NOT cuDNN for CUDA 13
sudo apt-get remove cudnn9-cuda-13
sudo apt-get -y install cudnn9-cuda-12
```

### Segmentation Fault during detection
**Solution**: Ensure GPU is available and drivers are properly installed:
```bash
nvidia-smi
```

## Performance

On NVIDIA RTX 4000 SFF Ada:
- YOLOv10n inference: ~5-10ms per image
- Model size: 9.9 MB
- Input: 640x640 RGB
- Output: Up to 8400 detections (post-processing filters by confidence)

## Next Steps

### Full Pipeline Integration
1. Add `FILTER_TYPE_YOLO` to proto definitions
2. Update `ProcessorEngine::ApplyFilters()` to handle YOLO
3. Add capability metadata to `ProcessorEngine::GetCapabilities()`
4. Implement result visualization (bounding boxes)

### Model Options
To use different YOLO models:
```bash
# YOLOv8n (stable, well-tested)
python scripts/models/export_yolo_to_onnx.py --model yolov8n

# YOLOv10s (better accuracy)
python scripts/models/export_yolo_to_onnx.py --model yolov10s

# YOLOv10m (balanced)
python scripts/models/export_yolo_to_onnx.py --model yolov10m
```

## References

- [ONNX Runtime C++ API](https://onnxruntime.ai/docs/api/c/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [cuDNN Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/)
