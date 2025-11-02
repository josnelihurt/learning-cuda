# Image Processing Kernels

> **Note: This file is archived for historical reference.**  
> All backlog items in this file have been migrated to GitHub Issues as part of the project's evolution from markdown-based backlog management to structured issue tracking. Each item was carefully analyzed, grouped with related tasks, and converted into actionable GitHub issues with proper labels, acceptance criteria, and context.
> 
> **Purpose**: This file is preserved to document the initial planning and evolution of the image processing kernel development roadmap. It serves as a historical record of how different filter categories and GPU programming techniques were organized for learning purposes.
>
> **Current Status**: All pending items have been converted to GitHub Issues (#512-519). The items are organized by filter family and complexity to guide the learning journey.
>
> **See**: [GitHub Issues](https://github.com/josnelihurt/learning-cuda/issues) for active project management.

Each kernel needs CPU + CUDA implementation, tests, and UI integration. Start with simpler ones (Box Blur, Sepia) before complex (Canny, CLAHE).

## Blur Filters

### Gaussian Blur
- [ ] #512 Separable convolution (horizontal + vertical)
- [ ] #512 Kernel sizes: 3x3, 5x5, 7x7
- [ ] #512 Sigma parameter
- [ ] #512 Shared memory optimization

### Box Blur
- [ ] #512 Simple averaging filter
- [ ] #512 Try prefix sum optimization (summed area table)

### Median Blur
- [ ] #512 Sliding window median
- [ ] #512 Parallel sorting or histogram approach

### Bilateral Filter
- [ ] #512 Edge-preserving blur
- [ ] #512 Spatial and range parameters

## Edge Detection

### Sobel
- [ ] #513 X and Y gradients
- [ ] #513 Magnitude and direction
- [ ] #513 Threshold parameter

### Scharr
- [ ] #513 Better rotational symmetry than Sobel

### Canny (Complex)
- [ ] #513 Gaussian blur
- [ ] #513 Gradient computation
- [ ] #513 Non-maximum suppression
- [ ] #513 Double threshold + hysteresis

### Laplacian of Gaussian
- [ ] #513 Second derivative
- [ ] #513 Zero crossings for edges

## Morphological Operations

### Erosion / Dilation
- [ ] #516 Binary and grayscale versions
- [ ] #516 Structuring elements (rectangle, ellipse, cross)

### Opening / Closing
- [ ] #516 Erosion + Dilation (opening)
- [ ] #516 Dilation + Erosion (closing)

### Morphological Gradient
- [ ] #516 Dilation - Erosion for edges

### Top-hat / Black-hat
- [ ] #516 Extract bright/dark features

## Color Space Conversions

### RGB to HSV
- [ ] #514 Implement color space transformation
- [ ] #514 Handle edge cases (undefined hue)
- [ ] #514 Optimize for parallel execution

**Learning Goals**: Color theory, conditional operations on GPU

### HSV to RGB
- [ ] #514 Implement inverse transformation
- [ ] #514 Test round-trip accuracy

### RGB to LAB
- [ ] #514 Implement perceptual color space conversion
- [ ] #514 Include XYZ intermediate step
- [ ] #514 Handle gamma correction

**Learning Goals**: Perceptual color spaces

### RGB to YCbCr
- [ ] #514 Implement for video processing
- [ ] #514 Useful for chroma subsampling

**Learning Goals**: Video color spaces

### White Balance Correction
- [ ] #514 Implement auto white balance
- [ ] #514 Add manual temperature adjustment
- [ ] #514 Use gray world or white patch algorithms

**Learning Goals**: Color correction techniques

## Phase 5: Sharpening & Enhancement

### Unsharp Masking
- [ ] #515 Blur image
- [ ] #515 Subtract from original to get mask
- [ ] #515 Add weighted mask back to original
- [ ] #515 Add strength parameter

**Learning Goals**: Multi-pass algorithms

### High-pass Filter
- [ ] #515 Implement frequency domain filtering
- [ ] #515 Or use Laplacian approximation

**Learning Goals**: Frequency domain concepts

### Histogram Equalization
- [ ] #515 Compute histogram
- [ ] #515 Calculate cumulative distribution function
- [ ] #515 Map pixel values
- [ ] #515 Optimize: Use atomic operations for histogram

**Learning Goals**: Histogram algorithms, atomic operations

### CLAHE (Contrast Limited Adaptive Histogram Equalization)
- [ ] #515 Divide image into tiles
- [ ] #515 Apply histogram equalization per tile
- [ ] #515 Interpolate border regions
- [ ] #515 Add clip limit parameter

**Learning Goals**: Tiled algorithms, complex pipelines

## Phase 6: Artistic Filters

### Sepia Tone
- [ ] #517 Apply sepia transformation matrix
- [ ] #517 Add intensity parameter

**Learning Goals**: Simple color transformations

### Vignette Effect
- [ ] #517 Create radial gradient mask
- [ ] #517 Darken edges
- [ ] #517 Add strength and radius parameters

**Learning Goals**: Distance-based effects

### Pixelation/Mosaic
- [ ] #517 Divide into blocks
- [ ] #517 Average color per block
- [ ] #517 Add block size parameter

**Learning Goals**: Block-based operations

### Oil Painting Effect
- [ ] #517 Implement Kuwahara filter variant
- [ ] #517 Add brush size parameter
- [ ] #517 Use neighborhood statistics

**Learning Goals**: Statistical filters

## Advanced Topics

### Kernel Fusion
- [ ] #518 Research combining multiple kernels into one launch
- [ ] #518 Implement grayscale + blur fusion example
- [ ] #518 Measure performance improvement

**Learning Goals**: GPU optimization, reducing memory transfers

### Pipeline Builder
- [ ] #519 Design DSL for filter composition
- [ ] #519 Auto-optimize pipeline order
- [ ] #519 Memory pooling for intermediate results

**Learning Goals**: API design, optimization strategies

### Memory Optimization
- [ ] #519 Implement memory pool for CUDA allocations
- [ ] #519 Reuse buffers across frames
- [ ] #519 Profile memory usage and transfers


## Notes

- Start with simpler filters (Box Blur, Sepia) before complex ones (Canny, CLAHE)
- Always implement CPU version first to validate algorithm
- Compare performance and correctness between CPU and CUDA
- Use existing grayscale implementation as reference
- Document kernel parameters and expected performance

