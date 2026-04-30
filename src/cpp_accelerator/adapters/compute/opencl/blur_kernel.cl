// Separable 5-tap Gaussian blur, sigma~1.0, weights = [1,4,6,4,1]/16
// Supports any number of channels (1, 3, or 4).
// Two-pass: horizontal then vertical, via an intermediate buffer.

__kernel void gaussian_blur_h(
    __global const uchar* restrict src,
    __global       uchar* restrict tmp,
    const int width,
    const int height,
    const int channels)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const float weights[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};

    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int k = -2; k <= 2; ++k) {
            int sx = clamp(x + k, 0, width - 1);
            sum += weights[k + 2] * (float)src[(y * width + sx) * channels + c];
        }
        tmp[(y * width + x) * channels + c] = (uchar)(sum + 0.5f);
    }
}

__kernel void gaussian_blur_v(
    __global const uchar* restrict tmp,
    __global       uchar* restrict dst,
    const int width,
    const int height,
    const int channels)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const float weights[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};

    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int k = -2; k <= 2; ++k) {
            int sy = clamp(y + k, 0, height - 1);
            sum += weights[k + 2] * (float)tmp[(sy * width + x) * channels + c];
        }
        dst[(y * width + x) * channels + c] = (uchar)(sum + 0.5f);
    }
}
