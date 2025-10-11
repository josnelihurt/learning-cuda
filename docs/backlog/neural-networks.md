# Neural Networks in CUDA

Learning path from basic CUDA kernels to production neural network frameworks.

## Learning Objectives

1. Understand neural network operations at CUDA kernel level
2. Implement backpropagation from scratch
3. Optimize matrix operations for GPU
4. Integrate production frameworks (cuDNN, TensorRT)
5. Deploy trained models for inference

## Phase 1: CUDA Building Blocks

### Matrix Operations

#### Basic Matrix Multiplication
- [ ] Implement naive matrix multiplication kernel
- [ ] Add tiling with shared memory optimization
- [ ] Implement matrix transpose
- [ ] Benchmark against cuBLAS

**Learning Goals**: Memory coalescing, shared memory, thread synchronization

#### Optimized Matrix Multiplication
- [ ] Implement register tiling
- [ ] Add bank conflict avoidance
- [ ] Try thread block size tuning
- [ ] Compare: naive vs tiled vs cuBLAS

**Learning Goals**: Advanced GPU optimization techniques

### Activation Functions

#### Forward Pass
- [ ] ReLU: `f(x) = max(0, x)`
- [ ] Sigmoid: `f(x) = 1 / (1 + e^-x)`
- [ ] Tanh: `f(x) = tanh(x)`
- [ ] Softmax: `f(x) = e^xi / Σ(e^xj)`
- [ ] Leaky ReLU: `f(x) = max(αx, x)`

**Learning Goals**: Element-wise operations, reduction operations (softmax)

#### Backward Pass (Derivatives)
- [ ] ReLU: `f'(x) = x > 0 ? 1 : 0`
- [ ] Sigmoid: `f'(x) = f(x) * (1 - f(x))`
- [ ] Tanh: `f'(x) = 1 - f(x)^2`
- [ ] Softmax: Jacobian matrix computation

**Learning Goals**: Gradient computation, chain rule

### Loss Functions

#### Forward Pass
- [ ] Mean Squared Error (MSE): `L = (1/n) * Σ(yi - ŷi)^2`
- [ ] Cross-Entropy: `L = -Σ(yi * log(ŷi))`
- [ ] Binary Cross-Entropy
- [ ] Hinge Loss (SVM)

#### Backward Pass
- [ ] MSE derivative: `∂L/∂ŷ = (2/n) * (ŷ - y)`
- [ ] Cross-Entropy derivative
- [ ] Combine with softmax (numerical stability)

**Learning Goals**: Loss computation, gradient flow

### Random Number Generation
- [ ] Initialize weights with Xavier/He initialization
- [ ] Use cuRAND library
- [ ] Implement dropout mask generation

**Learning Goals**: GPU random number generation

## Phase 2: Simple Networks

### Perceptron (Single Layer)

#### Implementation Tasks
- [ ] Create `Perceptron` class
- [ ] Forward pass: `y = Wx + b`
- [ ] Backward pass: compute gradients
- [ ] Update weights: `W = W - lr * ∂L/∂W`
- [ ] Train on linearly separable dataset

**Dataset**: Synthetic 2D points (XOR-like, but separable)


### Multi-Layer Perceptron (MLP)

#### Architecture
- Input layer: 784 neurons (28x28 MNIST)
- Hidden layer: 128 neurons + ReLU
- Output layer: 10 neurons + Softmax

#### Implementation Tasks
- [ ] Create `MLP` class with configurable layers
- [ ] Forward pass through all layers
- [ ] Backward pass with backpropagation
- [ ] Implement mini-batch gradient descent
- [ ] Add learning rate and momentum
- [ ] Train on MNIST dataset

**Learning Goals**: Backpropagation algorithm, multi-layer networks

**Expected Performance**: ~95% accuracy on MNIST

### Data Pipeline

#### MNIST Loader
- [ ] Download MNIST dataset
- [ ] Parse IDX file format
- [ ] Normalize pixel values (0-1)
- [ ] Create batches
- [ ] Shuffle data between epochs

#### Memory Management
- [ ] Pre-allocate GPU buffers
- [ ] Overlap data transfer with computation
- [ ] Use CUDA streams for pipelining

**Learning Goals**: Efficient data loading, GPU memory management

## Phase 3: Convolutional Neural Networks

### Convolution Operations

#### Naive Implementation
- [ ] Implement 2D convolution kernel
- [ ] Support padding and stride
- [ ] Handle multiple input/output channels

**Learning Goals**: Convolution algorithm

#### Im2col Implementation
- [ ] Transform image patches to columns
- [ ] Use optimized matrix multiplication (cuBLAS)
- [ ] Benchmark vs naive implementation

**Learning Goals**: Algorithm transformation, leveraging existing libraries

#### Optimized Implementation
- [ ] Use Winograd algorithm for 3x3 kernels
- [ ] Or integrate cuDNN convolution (see Phase 4)

#### Backward Pass
- [ ] Compute gradient w.r.t. input
- [ ] Compute gradient w.r.t. weights
- [ ] Handle padding and stride in gradients

### Pooling Operations

#### Max Pooling
- [ ] Forward pass: select maximum in each window
- [ ] Backward pass: route gradient to max location
- [ ] Support different pool sizes and strides

#### Average Pooling
- [ ] Forward pass: average over window
- [ ] Backward pass: distribute gradient equally

**Learning Goals**: Downsampling operations

### Batch Normalization

- [ ] Compute batch mean and variance
- [ ] Normalize: `x̂ = (x - μ) / √(σ² + ε)`
- [ ] Scale and shift: `y = γx̂ + β`
- [ ] Backward pass: gradient computation
- [ ] Track running statistics for inference

**Learning Goals**: Normalization techniques, training vs inference modes

### Simple CNN Architecture

#### LeNet-5 Style Network
```
Input (28x28x1)
  ↓
Conv 5x5x6 + ReLU
  ↓
MaxPool 2x2
  ↓
Conv 5x5x16 + ReLU
  ↓
MaxPool 2x2
  ↓
Flatten
  ↓
FC 120 + ReLU
  ↓
FC 84 + ReLU
  ↓
FC 10 + Softmax
```

#### Implementation
- [ ] Implement full CNN architecture
- [ ] Train on MNIST or CIFAR-10
- [ ] Achieve >98% accuracy on MNIST
- [ ] Visualize learned filters

**Learning Goals**: End-to-end CNN training

## Phase 4: Optimization & Training Infrastructure

### Optimizers

#### Stochastic Gradient Descent (SGD)
- [ ] Basic: `θ = θ - lr * ∇θ`
- [ ] With momentum: `v = βv + ∇θ; θ = θ - lr * v`
- [ ] With Nesterov momentum

#### Adam Optimizer
- [ ] First moment: `m = β₁m + (1-β₁)∇θ`
- [ ] Second moment: `v = β₂v + (1-β₂)∇θ²`
- [ ] Bias correction
- [ ] Update: `θ = θ - lr * m̂ / (√v̂ + ε)`

#### RMSprop
- [ ] Moving average of squared gradients
- [ ] Adaptive learning rate per parameter

**Learning Goals**: Advanced optimization algorithms

### Learning Rate Scheduling

- [ ] Step decay: reduce LR every N epochs
- [ ] Exponential decay: `lr = lr₀ * e^(-kt)`
- [ ] Cosine annealing
- [ ] Warmup strategy

### Regularization

- [ ] L2 weight decay: add `λ||W||²` to loss
- [ ] Dropout: randomly zero activations
- [ ] Data augmentation (for images)

### Training Loop
- [ ] Implement training loop with epochs
- [ ] Add validation set evaluation
- [ ] Early stopping based on validation loss
- [ ] Model checkpointing (save best model)
- [ ] Logging and metrics tracking

## Phase 5: Production Frameworks

### cuDNN Integration

#### Setup
- [ ] Install cuDNN library
- [ ] Add to Bazel dependencies
- [ ] Create C++ wrapper classes

#### Replace Custom Kernels
- [ ] Use `cudnnConvolutionForward` for convolution
- [ ] Use `cudnnActivationForward` for activations
- [ ] Use `cudnnPoolingForward` for pooling
- [ ] Use `cudnnBatchNormalizationForward` for batch norm

#### Benchmark
- [ ] Compare performance: custom kernels vs cuDNN
- [ ] Measure speedup (expect 2-10x for convolution)
- [ ] Profile memory usage

**Learning Goals**: Industry-standard library usage, performance optimization

**Resources**:
- https://developer.nvidia.com/cudnn
- https://docs.nvidia.com/deeplearning/cudnn/

### TensorRT for Inference

#### Model Export
- [ ] Save trained weights from custom network
- [ ] Or train in PyTorch and export to ONNX

#### TensorRT Pipeline
- [ ] Parse ONNX model with TensorRT
- [ ] Build optimized engine (fusion, precision calibration)
- [ ] Run inference with TensorRT runtime
- [ ] Benchmark vs custom implementation

#### Optimizations
- [ ] INT8 quantization for faster inference
- [ ] Layer fusion (conv + bias + relu → single kernel)
- [ ] Dynamic batching

**Learning Goals**: Deployment optimization, quantization

**Resources**:
- https://developer.nvidia.com/tensorrt
- https://github.com/NVIDIA/TensorRT

### ONNX Runtime with CUDA

#### Setup
- [ ] Install ONNX Runtime with CUDA execution provider
- [ ] Load ONNX model files

#### Integration
- [ ] Create inference service
- [ ] Run PyTorch/TensorFlow models
- [ ] Add to image processing pipeline

**Use Case**: Run pre-trained models (e.g., style transfer, super-resolution)

## Phase 6: Advanced Applications

### Image Classification Service

- [ ] Integrate trained CNN
- [ ] Add REST/gRPC endpoint for classification
- [ ] Return top-K predictions with confidence
- [ ] Visualize class activation maps (Grad-CAM)

### Style Transfer

- [ ] Implement neural style transfer
- [ ] Use VGG network for feature extraction
- [ ] Optimize content and style loss
- [ ] Real-time style transfer with fast networks

### Super-Resolution

- [ ] Implement SRCNN or ESPCN
- [ ] Upscale images 2x or 4x
- [ ] Compare with bicubic interpolation

### Object Detection (Advanced)

- [ ] Study YOLO or SSD architectures
- [ ] Implement bounding box regression
- [ ] Non-maximum suppression (NMS)
- [ ] Train on Pascal VOC or COCO

## Learning Resources

### Books
- "Programming Massively Parallel Processors" by Kirk & Hwu
- "Deep Learning" by Goodfellow, Bengio, Courville
- "Efficient Processing of Deep Neural Networks" by Sze et al.

### Online Courses
- Stanford CS231n: Convolutional Neural Networks
- Fast.ai: Practical Deep Learning
- NVIDIA Deep Learning Institute

### Papers
- ImageNet Classification with Deep CNNs (AlexNet)
- Very Deep CNNs for Large-Scale Recognition (VGG)
- Deep Residual Learning (ResNet)
- You Only Look Once (YOLO)

### Code References
- https://github.com/NVIDIA/cutlass (CUDA Templates for Linear Algebra)
- https://github.com/pytorch/pytorch (PyTorch CUDA kernels)
- https://github.com/microsoft/onnxruntime


## Notes

- Start simple: single layer perceptron before CNNs
- Always verify with CPU implementation first
- Use small datasets initially for fast iteration
- Profile everything - GPU performance can be surprising
- Compare with PyTorch/TensorFlow to validate correctness
- Focus on learning, not beating production frameworks

