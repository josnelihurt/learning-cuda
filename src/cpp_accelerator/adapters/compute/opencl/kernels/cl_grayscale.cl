// BT.601 grayscale: L = 0.299R + 0.587G + 0.114B
// Input:  packed RGB, 3 bytes/pixel
// Output: packed gray, 1 byte/pixel
__kernel void grayscale_bt601(
    __global const uchar* restrict in_rgb,
    __global       uchar* restrict out_gray,
    const int width,
    const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int pixel = y * width + x;
    uchar r = in_rgb[pixel * 3 + 0];
    uchar g = in_rgb[pixel * 3 + 1];
    uchar b = in_rgb[pixel * 3 + 2];

    float gray = 0.299f * (float)r + 0.587f * (float)g + 0.114f * (float)b;
    out_gray[pixel] = (uchar)(gray + 0.5f);
}
