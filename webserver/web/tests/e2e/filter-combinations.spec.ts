import { test, expect } from '@playwright/test';
import { TestHelpers } from './helpers/test-helpers';

test.describe('Filter Combinations', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
    await helpers.waitForPageReady();
    await helpers.enableDebugLogging();
    
    const count = await helpers.getSourceCount();
    if (count === 0) {
      await helpers.addSource('lena');
    }
    await helpers.selectSource(1);
  });

  test('Test Combo 1.1: Enable Both Grayscale and Blur Filters', async ({ page }) => {
    await helpers.enableFilter('grayscale');
    await helpers.enableFilter('blur');
    await page.waitForTimeout(1000);
    
    const grayscaleEnabled = await helpers.isFilterEnabled('grayscale');
    const blurEnabled = await helpers.isFilterEnabled('blur');
    
    expect(grayscaleEnabled).toBe(true);
    expect(blurEnabled).toBe(true);
  });

  test('Test Combo 1.2: Apply Grayscale Then Blur', async ({ page }) => {
    await helpers.enableFilter('grayscale');
    await helpers.selectFilterParameter('grayscale', 'algorithm', 'bt601');
    await page.waitForTimeout(500);
    
    await helpers.enableFilter('blur');
    await helpers.setBlurParameter('blur', 'kernel_size', '5');
    await helpers.setBlurParameter('blur', 'sigma', '1.0');
    await page.waitForTimeout(1000);
    
    const grayscaleEnabled = await helpers.isFilterEnabled('grayscale');
    const blurEnabled = await helpers.isFilterEnabled('blur');
    const algorithm = await helpers.getSelectedParameter('grayscale', 'algorithm');
    const kernelSize = await helpers.getBlurParameter('blur', 'kernel_size');
    
    expect(grayscaleEnabled).toBe(true);
    expect(blurEnabled).toBe(true);
    expect(algorithm).toBe('bt601');
    expect(kernelSize).toBe('5');
  });

  test('Test Combo 1.3: Apply Blur Then Grayscale', async ({ page }) => {
    await helpers.enableFilter('blur');
    await helpers.setBlurParameter('blur', 'kernel_size', '7');
    await helpers.setBlurParameter('blur', 'sigma', '1.5');
    await page.waitForTimeout(500);
    
    await helpers.enableFilter('grayscale');
    await helpers.selectFilterParameter('grayscale', 'algorithm', 'bt709');
    await page.waitForTimeout(1000);
    
    const grayscaleEnabled = await helpers.isFilterEnabled('grayscale');
    const blurEnabled = await helpers.isFilterEnabled('blur');
    const algorithm = await helpers.getSelectedParameter('grayscale', 'algorithm');
    const kernelSize = await helpers.getBlurParameter('blur', 'kernel_size');
    
    expect(grayscaleEnabled).toBe(true);
    expect(blurEnabled).toBe(true);
    expect(algorithm).toBe('bt709');
    expect(kernelSize).toBe('7');
  });

  test('Test Combo 2.1: Change Grayscale Algorithm While Blur Active', async ({ page }) => {
    await helpers.enableFilter('grayscale');
    await helpers.enableFilter('blur');
    await page.waitForTimeout(500);
    
    const algorithms = ['bt601', 'bt709', 'average', 'luminosity'];
    
    for (const algo of algorithms) {
      await helpers.selectFilterParameter('grayscale', 'algorithm', algo);
      await page.waitForTimeout(500);
      
      const selected = await helpers.getSelectedParameter('grayscale', 'algorithm');
      expect(selected).toBe(algo);
      
      const blurStillEnabled = await helpers.isFilterEnabled('blur');
      expect(blurStillEnabled).toBe(true);
    }
  });

  test('Test Combo 2.2: Change Blur Parameters While Grayscale Active', async ({ page }) => {
    await helpers.enableFilter('grayscale');
    await helpers.enableFilter('blur');
    await page.waitForTimeout(500);
    
    const kernelSizes = ['5', '7', '9'];
    const sigmaValues = ['1.0', '1.5', '2.0'];
    
    for (const size of kernelSizes) {
      await helpers.setBlurParameter('blur', 'kernel_size', size);
      await page.waitForTimeout(300);
      
      const currentSize = await helpers.getBlurParameter('blur', 'kernel_size');
      expect(currentSize).toBe(size);
      
      const grayscaleStillEnabled = await helpers.isFilterEnabled('grayscale');
      expect(grayscaleStillEnabled).toBe(true);
    }
    
    for (const sigma of sigmaValues) {
      await helpers.setBlurParameter('blur', 'sigma', sigma);
      await page.waitForTimeout(300);
      
      const currentSigma = await helpers.getBlurParameter('blur', 'sigma');
      expect(parseFloat(currentSigma || '0')).toBeCloseTo(parseFloat(sigma), 1);
    }
  });

  test('Test Combo 3.1: Disable Grayscale While Blur Active', async ({ page }) => {
    await helpers.enableFilter('grayscale');
    await helpers.enableFilter('blur');
    await page.waitForTimeout(500);
    
    await helpers.disableFilter('grayscale');
    await page.waitForTimeout(500);
    
    const grayscaleEnabled = await helpers.isFilterEnabled('grayscale');
    const blurEnabled = await helpers.isFilterEnabled('blur');
    
    expect(grayscaleEnabled).toBe(false);
    expect(blurEnabled).toBe(true);
  });

  test('Test Combo 3.2: Disable Blur While Grayscale Active', async ({ page }) => {
    await helpers.enableFilter('grayscale');
    await helpers.enableFilter('blur');
    await page.waitForTimeout(500);
    
    await helpers.disableFilter('blur');
    await page.waitForTimeout(500);
    
    const grayscaleEnabled = await helpers.isFilterEnabled('grayscale');
    const blurEnabled = await helpers.isFilterEnabled('blur');
    
    expect(grayscaleEnabled).toBe(true);
    expect(blurEnabled).toBe(false);
  });

  test('Test Combo 4.1: Change Resolution With Both Filters Active', async ({ page }) => {
    await helpers.enableFilter('grayscale');
    await helpers.enableFilter('blur');
    await page.waitForTimeout(500);
    
    const resolutions = ['original', 'half', 'quarter', 'original'];
    
    for (const resolution of resolutions) {
      helpers.clearConsoleLogs();
      await helpers.selectResolution(resolution);
      await page.waitForTimeout(500);
      
      const grayscaleEnabled = await helpers.isFilterEnabled('grayscale');
      const blurEnabled = await helpers.isFilterEnabled('blur');
      
      expect(grayscaleEnabled).toBe(true);
      expect(blurEnabled).toBe(true);
    }
  });

  test('Test Combo 5.1: Multiple Sources With Different Filter Combinations', async ({ page }) => {
    for (let i = 0; i < 3; i++) {
      await helpers.addSource('lena');
    }
    
    await helpers.selectSource(1);
    await helpers.enableFilter('grayscale');
    await helpers.selectFilterParameter('grayscale', 'algorithm', 'bt601');
    await page.waitForTimeout(300);
    
    await helpers.selectSource(2);
    await helpers.enableFilter('blur');
    await helpers.setBlurParameter('blur', 'kernel_size', '5');
    await page.waitForTimeout(300);
    
    await helpers.selectSource(3);
    await helpers.enableFilter('grayscale');
    await helpers.enableFilter('blur');
    await helpers.selectFilterParameter('grayscale', 'algorithm', 'bt709');
    await helpers.setBlurParameter('blur', 'kernel_size', '7');
    await page.waitForTimeout(300);
    
    await helpers.selectSource(1);
    await page.waitForTimeout(200);
    expect(await helpers.isFilterEnabled('grayscale')).toBe(true);
    expect(await helpers.isFilterEnabled('blur')).toBe(false);
    expect(await helpers.getSelectedParameter('grayscale', 'algorithm')).toBe('bt601');
    
    await helpers.selectSource(2);
    await page.waitForTimeout(200);
    expect(await helpers.isFilterEnabled('grayscale')).toBe(false);
    expect(await helpers.isFilterEnabled('blur')).toBe(true);
    expect(await helpers.getBlurParameter('blur', 'kernel_size')).toBe('5');
    
    await helpers.selectSource(3);
    await page.waitForTimeout(200);
    expect(await helpers.isFilterEnabled('grayscale')).toBe(true);
    expect(await helpers.isFilterEnabled('blur')).toBe(true);
    expect(await helpers.getSelectedParameter('grayscale', 'algorithm')).toBe('bt709');
    expect(await helpers.getBlurParameter('blur', 'kernel_size')).toBe('7');
  });

  test('Test Combo 6.1: Re-enable Filters After Disabling Both', async ({ page }) => {
    await helpers.enableFilter('grayscale');
    await helpers.enableFilter('blur');
    await page.waitForTimeout(500);
    
    await helpers.disableFilter('grayscale');
    await helpers.disableFilter('blur');
    await page.waitForTimeout(500);
    
    expect(await helpers.isFilterEnabled('grayscale')).toBe(false);
    expect(await helpers.isFilterEnabled('blur')).toBe(false);
    
    await helpers.enableFilter('grayscale');
    await helpers.enableFilter('blur');
    await page.waitForTimeout(500);
    
    expect(await helpers.isFilterEnabled('grayscale')).toBe(true);
    expect(await helpers.isFilterEnabled('blur')).toBe(true);
  });
});
