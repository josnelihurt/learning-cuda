# Image Processing Kernels

Each kernel needs CPU + CUDA implementation, tests, and UI integration. Start with simpler ones (Box Blur, Sepia) before complex (Canny, CLAHE).

## Blur Filters

### Gaussian Blur
- [ ] Separable convolution (horizontal + vertical)
- [ ] Kernel sizes: 3x3, 5x5, 7x7
- [ ] Sigma parameter
- [ ] Shared memory optimization

### Box Blur
- [ ] Simple averaging filter
- [ ] Try prefix sum optimization (summed area table)

### Median Blur
- [ ] Sliding window median
- [ ] Parallel sorting or histogram approach

### Bilateral Filter
- [ ] Edge-preserving blur
- [ ] Spatial and range parameters

## Edge Detection

### Sobel
- [ ] X and Y gradients
- [ ] Magnitude and direction
- [ ] Threshold parameter

### Scharr
- [ ] Better rotational symmetry than Sobel

### Canny (Complex)
- [ ] Gaussian blur
- [ ] Gradient computation
- [ ] Non-maximum suppression
- [ ] Double threshold + hysteresis

### Laplacian of Gaussian
- [ ] Second derivative
- [ ] Zero crossings for edges

## Morphological Operations

### Erosion / Dilation
- [ ] Binary and grayscale versions
- [ ] Structuring elements (rectangle, ellipse, cross)

### Opening / Closing
- [ ] Erosion + Dilation (opening)
- [ ] Dilation + Erosion (closing)

### Morphological Gradient
- [ ] Dilation - Erosion for edges

### Top-hat / Black-hat
- [ ] Extract bright/dark features

## Color Space Conversions

### RGB to HSV
- [ ] Implement color space transformation
- [ ] Handle edge cases (undefined hue)
- [ ] Optimize for parallel execution

**Learning Goals**: Color theory, conditional operations on GPU

### HSV to RGB
- [ ] Implement inverse transformation
- [ ] Test round-trip accuracy

### RGB to LAB
- [ ] Implement perceptual color space conversion
- [ ] Include XYZ intermediate step
- [ ] Handle gamma correction

**Learning Goals**: Perceptual color spaces

### RGB to YCbCr
- [ ] Implement for video processing
- [ ] Useful for chroma subsampling

**Learning Goals**: Video color spaces

### White Balance Correction
- [ ] Implement auto white balance
- [ ] Add manual temperature adjustment
- [ ] Use gray world or white patch algorithms

**Learning Goals**: Color correction techniques

## Phase 5: Sharpening & Enhancement

### Unsharp Masking
- [ ] Blur image
- [ ] Subtract from original to get mask
- [ ] Add weighted mask back to original
- [ ] Add strength parameter

**Learning Goals**: Multi-pass algorithms

### High-pass Filter
- [ ] Implement frequency domain filtering
- [ ] Or use Laplacian approximation

**Learning Goals**: Frequency domain concepts

### Histogram Equalization
- [ ] Compute histogram
- [ ] Calculate cumulative distribution function
- [ ] Map pixel values
- [ ] Optimize: Use atomic operations for histogram

**Learning Goals**: Histogram algorithms, atomic operations

### CLAHE (Contrast Limited Adaptive Histogram Equalization)
- [ ] Divide image into tiles
- [ ] Apply histogram equalization per tile
- [ ] Interpolate border regions
- [ ] Add clip limit parameter

**Learning Goals**: Tiled algorithms, complex pipelines

## Phase 6: Artistic Filters

### Sepia Tone
- [ ] Apply sepia transformation matrix
- [ ] Add intensity parameter

**Learning Goals**: Simple color transformations

### Vignette Effect
- [ ] Create radial gradient mask
- [ ] Darken edges
- [ ] Add strength and radius parameters

**Learning Goals**: Distance-based effects

### Pixelation/Mosaic
- [ ] Divide into blocks
- [ ] Average color per block
- [ ] Add block size parameter

**Learning Goals**: Block-based operations

### Oil Painting Effect
- [ ] Implement Kuwahara filter variant
- [ ] Add brush size parameter
- [ ] Use neighborhood statistics

**Learning Goals**: Statistical filters

## Advanced Topics

### Kernel Fusion
- [ ] Research combining multiple kernels into one launch
- [ ] Implement grayscale + blur fusion example
- [ ] Measure performance improvement

**Learning Goals**: GPU optimization, reducing memory transfers

### Pipeline Builder
- [ ] Design DSL for filter composition
- [ ] Auto-optimize pipeline order
- [ ] Memory pooling for intermediate results

**Learning Goals**: API design, optimization strategies

### Memory Optimization
- [ ] Implement memory pool for CUDA allocations
- [ ] Reuse buffers across frames
- [ ] Profile memory usage and transfers


## Notes

- Start with simpler filters (Box Blur, Sepia) before complex ones (Canny, CLAHE)
- Always implement CPU version first to validate algorithm
- Compare performance and correctness between CPU and CUDA
- Use existing grayscale implementation as reference
- Document kernel parameters and expected performance

