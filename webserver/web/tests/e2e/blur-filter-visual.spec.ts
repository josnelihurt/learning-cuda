import { test, expect } from '@playwright/test';
import { TestHelpers } from './helpers/test-helpers';
import * as crypto from 'crypto';

test.describe('Blur Filter Visual Validation', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    const count = await helpers.getSourceCount();
    if (count === 0) {
      await helpers.addSource('lena');
    }
    await helpers.selectSource(1);
    await page.waitForTimeout(1000);
  });

  test('CRITICAL: Blur filter applies correctly to image (no white screen)', async ({ page }) => {
    // Wait for video-grid to load
    await page.waitForSelector('video-grid', { state: 'visible', timeout: 10000 });
    await page.waitForTimeout(2000);

    // Step 1: Capture image bytes BEFORE applying blur
    const bytesBefore = await page.evaluate(() => {
      return new Promise<number[] | null>((resolve) => {
        const grid = document.querySelector('video-grid');
        if (!grid?.shadowRoot) {
          console.error('[TEST] video-grid or shadowRoot not found');
          resolve(null);
          return;
        }

        const cards = grid.shadowRoot.querySelectorAll('video-source-card');
        if (cards.length === 0) {
          console.error('[TEST] No video-source-card found');
          resolve(null);
          return;
        }

        const videoCard = cards[0];
        if (!videoCard?.shadowRoot) {
          console.error('[TEST] video-source-card shadowRoot not found');
          resolve(null);
          return;
        }

        const img = videoCard.shadowRoot.querySelector('img') as HTMLImageElement;
        if (!img) {
          console.error('[TEST] Image not found in shadow DOM');
          resolve(null);
          return;
        }

        const captureBytes = () => {
          const width = img.naturalWidth || img.width || 512;
          const height = img.naturalHeight || img.height || 512;
          
          if (width === 0 || height === 0) {
            console.error('[TEST] Image has zero dimensions:', width, height);
            resolve(null);
            return;
          }

          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d', { willReadFrequently: true });
          if (!ctx) {
            console.error('[TEST] Canvas context not available');
            resolve(null);
            return;
          }

          try {
            ctx.drawImage(img, 0, 0);
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            console.log('[TEST] Captured bytes before blur:', imageData.data.length);
            resolve(Array.from(imageData.data));
          } catch (e) {
            console.error('[TEST] Error capturing bytes:', e);
            resolve(null);
          }
        };

        if (img.complete && (img.naturalWidth > 0 || img.width > 0)) {
          captureBytes();
        } else {
          img.onload = captureBytes;
          img.onerror = () => {
            console.error('[TEST] Image load error');
            resolve(null);
          };
          setTimeout(() => {
            if (img.complete && (img.naturalWidth > 0 || img.width > 0)) {
              captureBytes();
            } else {
              console.error('[TEST] Image timeout - naturalWidth:', img.naturalWidth, 'width:', img.width);
              resolve(null);
            }
          }, 3000);
        }
      });
    });

    expect(bytesBefore).not.toBeNull();
    expect(bytesBefore!.length).toBeGreaterThan(0);
    console.log('[TEST] Bytes before blur:', bytesBefore!.length, 'bytes');

    // Step 2: Apply blur filter (WITHOUT grayscale)
    await helpers.enableFilter('blur');
    await page.waitForTimeout(500);
    await helpers.setBlurParameter('blur', 'kernel_size', '5');
    await helpers.setBlurParameter('blur', 'sigma', '1.0');
    await helpers.selectFilterParameter('blur', 'border_mode', 'REFLECT');
    await helpers.setBlurParameter('blur', 'separable', 'true');

    // Step 3: Wait for image to update (allow time for WebSocket communication and processing)
    await page.waitForTimeout(3000);

    // Step 4: Capture image bytes AFTER applying blur
    await page.waitForTimeout(2000);
    
    const bytesAfter = await page.evaluate(() => {
      return new Promise<number[] | null>((resolve) => {
        const grid = document.querySelector('video-grid');
        if (!grid?.shadowRoot) {
          console.error('[TEST] video-grid or shadowRoot not found after blur');
          resolve(null);
          return;
        }

        const cards = grid.shadowRoot.querySelectorAll('video-source-card');
        if (cards.length === 0) {
          console.error('[TEST] No video-source-card found after blur');
          resolve(null);
          return;
        }

        const videoCard = cards[0];
        if (!videoCard?.shadowRoot) {
          console.error('[TEST] video-source-card shadowRoot not found after blur');
          resolve(null);
          return;
        }

        const img = videoCard.shadowRoot.querySelector('img') as HTMLImageElement;
        if (!img) {
          console.error('[TEST] Image not found after blur');
          resolve(null);
          return;
        }

        const captureBytes = () => {
          const width = img.naturalWidth || img.width || 512;
          const height = img.naturalHeight || img.height || 512;
          
          if (width === 0 || height === 0) {
            console.error('[TEST] Image has zero dimensions after blur:', width, height);
            resolve(null);
            return;
          }

          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d', { willReadFrequently: true });
          if (!ctx) {
            console.error('[TEST] Canvas context not available after blur');
            resolve(null);
            return;
          }

          try {
            ctx.drawImage(img, 0, 0);
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            console.log('[TEST] Captured bytes after blur:', imageData.data.length);
            resolve(Array.from(imageData.data));
          } catch (e) {
            console.error('[TEST] Error capturing bytes after blur:', e);
            resolve(null);
          }
        };

        if (img.complete && (img.naturalWidth > 0 || img.width > 0)) {
          captureBytes();
        } else {
          img.onload = captureBytes;
          img.onerror = () => {
            console.error('[TEST] Image load error after blur');
            resolve(null);
          };
          setTimeout(() => {
            if (img.complete && (img.naturalWidth > 0 || img.width > 0)) {
              captureBytes();
            } else {
              console.error('[TEST] Image timeout after blur - naturalWidth:', img.naturalWidth, 'width:', img.width);
              resolve(null);
            }
          }, 3000);
        }
      });
    });

    expect(bytesAfter).not.toBeNull();
    expect(bytesAfter!.length).toBeGreaterThan(0);
    console.log('[TEST] Bytes after blur:', bytesAfter!.length, 'bytes');

    // Step 5: Verify image dimensions match
    expect(bytesBefore!.length).toBe(bytesAfter!.length);
    const totalPixels = bytesBefore!.length / 4;

    // Step 6: Calculate pixel differences
    let differentPixels = 0;
    let whitePixels = 0;
    let uniformPixels = 0;
    const pixelVariation = new Map<string, number>();

    const firstPixelR = bytesAfter![0];
    const firstPixelG = bytesAfter![1];
    const firstPixelB = bytesAfter![2];

    for (let i = 0; i < bytesBefore!.length; i += 4) {
      const r1 = bytesBefore![i];
      const g1 = bytesBefore![i + 1];
      const b1 = bytesBefore![i + 2];

      const r2 = bytesAfter![i];
      const g2 = bytesAfter![i + 1];
      const b2 = bytesAfter![i + 2];

      // Check if pixel is white (all channels > 250)
      if (r2 > 250 && g2 > 250 && b2 > 250) {
        whitePixels++;
      }

      // Check if pixel matches first pixel (uniformity check)
      if (r2 === firstPixelR && g2 === firstPixelG && b2 === firstPixelB) {
        uniformPixels++;
      }

      // Track pixel value variation
      const pixelKey = `${r2},${g2},${b2}`;
      pixelVariation.set(pixelKey, (pixelVariation.get(pixelKey) || 0) + 1);

      // Consider pixel different if any RGB channel differs by more than threshold
      const threshold = 2;
      if (Math.abs(r1 - r2) > threshold || Math.abs(g1 - g2) > threshold || Math.abs(b1 - b2) > threshold) {
        differentPixels++;
      }
    }

    const percentDifferent = (differentPixels / totalPixels) * 100;
    const percentWhite = (whitePixels / totalPixels) * 100;
    const percentUniform = (uniformPixels / totalPixels) * 100;
    const uniqueColors = pixelVariation.size;

    console.log(`\n=== BLUR FILTER VISUAL VALIDATION ===`);
    console.log(`Total pixels: ${totalPixels.toLocaleString()}`);
    console.log(`Different pixels: ${differentPixels.toLocaleString()} (${percentDifferent.toFixed(2)}%)`);
    console.log(`White pixels: ${whitePixels.toLocaleString()} (${percentWhite.toFixed(2)}%)`);
    console.log(`Uniform pixels: ${uniformPixels.toLocaleString()} (${percentUniform.toFixed(2)}%)`);
    console.log(`Unique colors: ${uniqueColors.toLocaleString()}`);

    // Calculate SHA-256 hashes for comparison
    // @ts-ignore - Buffer is available in Playwright test environment
    const hashBefore = crypto.createHash('sha256').update(Buffer.from(bytesBefore!)).digest('hex');
    // @ts-ignore - Buffer is available in Playwright test environment
    const hashAfter = crypto.createHash('sha256').update(Buffer.from(bytesAfter!)).digest('hex');

    console.log(`Hash before: ${hashBefore.substring(0, 16)}...`);
    console.log(`Hash after: ${hashAfter.substring(0, 16)}...`);
    console.log('');

    // Step 7: Critical validations
    // Validation 1: Image should have changed (blur applied)
    expect(percentDifferent).toBeGreaterThan(0.1, 
      'Blur filter should cause at least 0.1% pixel changes');
    
    // Validation 2: Image should NOT be mostly white (white screen bug)
    expect(percentWhite).toBeLessThan(50,
      'Image should not be mostly white (white screen bug detected)');
    
    // Validation 3: Image should have color variation (not uniform)
    expect(percentUniform).toBeLessThan(50,
      'Image should not be uniform (all pixels same color)');
    
    // Validation 4: Image should have many unique colors (not corrupted)
    expect(uniqueColors).toBeGreaterThan(100,
      'Image should have many unique colors (not corrupted)');
    
    // Validation 5: Hashes should be different (image changed)
    expect(hashBefore).not.toBe(hashAfter,
      'Image hash should change after applying blur filter');

    console.log('All validations passed - Blur filter is working correctly!');
  });
});
