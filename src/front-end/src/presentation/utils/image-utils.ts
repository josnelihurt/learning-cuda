export function frameResponseToDataUrl(
  bytes: Uint8Array,
  width: number,
  height: number,
  channels: number
): string {
  const rgba = new Uint8ClampedArray(width * height * 4);

  if (channels === 1) {
    for (let index = 0; index < width * height; index += 1) {
      const value = bytes[index] ?? 0;
      const offset = index * 4;
      rgba[offset] = value;
      rgba[offset + 1] = value;
      rgba[offset + 2] = value;
      rgba[offset + 3] = 255;
    }
  } else if (channels === 3) {
    for (let index = 0; index < width * height; index += 1) {
      const srcOffset = index * 3;
      const dstOffset = index * 4;
      rgba[dstOffset] = bytes[srcOffset] ?? 0;
      rgba[dstOffset + 1] = bytes[srcOffset + 1] ?? 0;
      rgba[dstOffset + 2] = bytes[srcOffset + 2] ?? 0;
      rgba[dstOffset + 3] = 255;
    }
  } else {
    for (let index = 0; index < width * height; index += 1) {
      const srcOffset = index * channels;
      const dstOffset = index * 4;
      rgba[dstOffset] = bytes[srcOffset] ?? 0;
      rgba[dstOffset + 1] = bytes[srcOffset + 1] ?? 0;
      rgba[dstOffset + 2] = bytes[srcOffset + 2] ?? 0;
      rgba[dstOffset + 3] = channels >= 4 ? (bytes[srcOffset + 3] ?? 255) : 255;
    }
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Canvas context is not available');
  }

  context.putImageData(new ImageData(rgba, width, height), 0, 0);
  return canvas.toDataURL('image/png');
}

export async function rasterizeImageToRgb(
  imageSrc: string,
  width: number,
  height: number
): Promise<Uint8Array> {
  const image = new Image();
  image.crossOrigin = 'anonymous';

  await new Promise<void>((resolve, reject) => {
    image.onload = () => resolve();
    image.onerror = () => reject(new Error('Failed to load original image'));
    image.src = imageSrc;
  });

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Canvas context is not available');
  }

  context.drawImage(image, 0, 0, width, height);
  const raster = context.getImageData(0, 0, width, height);
  const rgb = new Uint8Array(width * height * 3);

  for (let index = 0, pixel = 0; index < raster.data.length; index += 4, pixel += 3) {
    rgb[pixel] = raster.data[index];
    rgb[pixel + 1] = raster.data[index + 1];
    rgb[pixel + 2] = raster.data[index + 2];
  }

  return rgb;
}
