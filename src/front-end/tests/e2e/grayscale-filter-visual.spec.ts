import { test, expect } from '@playwright/test';
import * as crypto from 'crypto';

function analyzePixels(bytes: number[]) {
  const totalPixels = bytes.length / 4;
  let grayscalePixels = 0;
  let whitePixels = 0;
  let uniformPixels = 0;
  let uniqueColors = 0;
  const colorSet = new Set<string>();

  const firstPixelR = bytes[0];
  const firstPixelG = bytes[1];
  const firstPixelB = bytes[2];

  for (let i = 0; i < bytes.length; i += 4) {
    const r = bytes[i];
    const g = bytes[i + 1];
    const b = bytes[i + 2];

    if (Math.abs(r - g) <= 2 && Math.abs(g - b) <= 2 && Math.abs(r - b) <= 2) {
      grayscalePixels++;
    }

    if (r > 250 && g > 250 && b > 250) {
      whitePixels++;
    }

    if (r === firstPixelR && g === firstPixelG && b === firstPixelB) {
      uniformPixels++;
    }

    colorSet.add(`${r},${g},${b}`);
  }

  uniqueColors = colorSet.size;

  return {
    totalPixels,
    grayscalePercent: (grayscalePixels / totalPixels) * 100,
    whitePercent: (whitePixels / totalPixels) * 100,
    uniformPercent: (uniformPixels / totalPixels) * 100,
    uniqueColors,
  };
}

test.describe('Grayscale Filter Visual Validation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForSelector('header', { timeout: 10000 });
    await page.waitForSelector('[data-testid="source-card-1"]', { timeout: 30000 });
    await page.waitForTimeout(1200);
    const skipButton = page.locator('button', { hasText: 'Skip' });
    if (await skipButton.isVisible().catch(() => false)) {
      await skipButton.click({ force: true });
      await expect(skipButton).toBeHidden({ timeout: 5000 });
    }
    await expect(page.locator('[data-testid="source-card-1"]')).toBeVisible();
    await page.waitForTimeout(1000);
  });

  test('applies grayscale to Lena and updates rendered pixels', async ({ page }) => {
    await page.waitForSelector('[data-testid="video-grid"]', { state: 'visible', timeout: 10000 });
    await page.waitForTimeout(2000);

    const bytesBefore = await page.evaluate(() => {
      return new Promise<number[] | null>((resolve) => {
        const card = document.querySelector('[data-testid="source-card-1"]');
        if (!card) {
          resolve(null);
          return;
        }

        const img = card.querySelector('img') as HTMLImageElement | null;
        if (!img) {
          resolve(null);
          return;
        }

        const captureBytes = () => {
          const width = img.naturalWidth || img.width || 512;
          const height = img.naturalHeight || img.height || 512;
          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d', { willReadFrequently: true });
          if (!ctx) {
            resolve(null);
            return;
          }

          ctx.drawImage(img, 0, 0);
          const imageData = ctx.getImageData(0, 0, width, height);
          resolve(Array.from(imageData.data));
        };

        if (img.complete && (img.naturalWidth > 0 || img.width > 0)) {
          captureBytes();
          return;
        }

        img.onload = captureBytes;
        img.onerror = () => resolve(null);
        setTimeout(() => {
          if (img.complete && (img.naturalWidth > 0 || img.width > 0)) {
            captureBytes();
          } else {
            resolve(null);
          }
        }, 3000);
      });
    });

    expect(bytesBefore).not.toBeNull();
    expect(bytesBefore!.length).toBeGreaterThan(0);

    const grayscaleCheckbox = page.locator('[data-testid="filter-checkbox-grayscale"]');
    await expect(grayscaleCheckbox).toBeVisible();
    if (!(await grayscaleCheckbox.isChecked())) {
      await grayscaleCheckbox.check();
    }
    await page.locator('[data-testid="filter-parameter-grayscale-algorithm-bt601"]').click();
    await page.waitForTimeout(3000);

    const bytesAfter = await page.evaluate(() => {
      return new Promise<number[] | null>((resolve) => {
        const card = document.querySelector('[data-testid="source-card-1"]');
        if (!card) {
          resolve(null);
          return;
        }

        const img = card.querySelector('img') as HTMLImageElement | null;
        if (!img) {
          resolve(null);
          return;
        }

        const captureBytes = () => {
          const width = img.naturalWidth || img.width || 512;
          const height = img.naturalHeight || img.height || 512;
          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d', { willReadFrequently: true });
          if (!ctx) {
            resolve(null);
            return;
          }

          ctx.drawImage(img, 0, 0);
          const imageData = ctx.getImageData(0, 0, width, height);
          resolve(Array.from(imageData.data));
        };

        if (img.complete && (img.naturalWidth > 0 || img.width > 0)) {
          captureBytes();
          return;
        }

        img.onload = captureBytes;
        img.onerror = () => resolve(null);
        setTimeout(() => {
          if (img.complete && (img.naturalWidth > 0 || img.width > 0)) {
            captureBytes();
          } else {
            resolve(null);
          }
        }, 3000);
      });
    });

    expect(bytesAfter).not.toBeNull();
    expect(bytesAfter!.length).toBe(bytesBefore!.length);

    let differentPixels = 0;
    for (let i = 0; i < bytesBefore!.length; i += 4) {
      const beforeR = bytesBefore![i];
      const beforeG = bytesBefore![i + 1];
      const beforeB = bytesBefore![i + 2];
      const afterR = bytesAfter![i];
      const afterG = bytesAfter![i + 1];
      const afterB = bytesAfter![i + 2];

      if (
        Math.abs(beforeR - afterR) > 2 ||
        Math.abs(beforeG - afterG) > 2 ||
        Math.abs(beforeB - afterB) > 2
      ) {
        differentPixels++;
      }
    }

    const beforeAnalysis = analyzePixels(bytesBefore!);
    const afterAnalysis = analyzePixels(bytesAfter!);
    const percentDifferent = (differentPixels / beforeAnalysis.totalPixels) * 100;

    const hashBefore = crypto.createHash('sha256').update(Buffer.from(bytesBefore!)).digest('hex');
    const hashAfter = crypto.createHash('sha256').update(Buffer.from(bytesAfter!)).digest('hex');

    console.log(`Before grayscale pixels: ${beforeAnalysis.grayscalePercent.toFixed(2)}%`);
    console.log(`After grayscale pixels: ${afterAnalysis.grayscalePercent.toFixed(2)}%`);
    console.log(`Different pixels: ${percentDifferent.toFixed(2)}%`);
    console.log(`White pixels after: ${afterAnalysis.whitePercent.toFixed(2)}%`);
    console.log(`Uniform pixels after: ${afterAnalysis.uniformPercent.toFixed(2)}%`);
    console.log(`Unique colors after: ${afterAnalysis.uniqueColors}`);

    expect(percentDifferent).toBeGreaterThan(1);
    expect(afterAnalysis.grayscalePercent).toBeGreaterThan(95);
    expect(afterAnalysis.grayscalePercent).toBeGreaterThan(beforeAnalysis.grayscalePercent + 20);
    expect(afterAnalysis.whitePercent).toBeLessThan(50);
    expect(afterAnalysis.uniformPercent).toBeLessThan(50);
    expect(afterAnalysis.uniqueColors).toBeGreaterThan(100);
    expect(hashBefore).not.toBe(hashAfter);
  });
});
