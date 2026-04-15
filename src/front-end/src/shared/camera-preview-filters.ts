export type CameraPreviewFilterState = {
  id: string;
  parameters: Record<string, string>;
};

function parsePositiveNumber(value: string | undefined): number | null {
  if (value === undefined) {
    return null;
  }
  const parsed = Number.parseFloat(value);
  if (Number.isNaN(parsed) || parsed < 0) {
    return null;
  }
  return parsed;
}

function toBlurPixels(parameters: Record<string, string>): number {
  const sigma = parsePositiveNumber(parameters.sigma);
  if (sigma !== null) {
    return Math.max(0, sigma);
  }

  const kernelSize = parsePositiveNumber(parameters.kernel_size);
  if (kernelSize !== null) {
    return Math.max(0, kernelSize / 2);
  }

  return 1;
}

export function buildCameraPreviewFilterValue(
  activeFilters: CameraPreviewFilterState[] | undefined
): string {
  if (!activeFilters || activeFilters.length === 0) {
    return 'none';
  }

  const cssFilters: string[] = [];

  for (const filter of activeFilters) {
    if (filter.id === 'none') {
      continue;
    }

    if (filter.id === 'grayscale') {
      cssFilters.push('grayscale(1)');
      continue;
    }

    if (filter.id === 'blur') {
      cssFilters.push(`blur(${toBlurPixels(filter.parameters).toFixed(2)}px)`);
    }
  }

  return cssFilters.length > 0 ? cssFilters.join(' ') : 'none';
}
