import { test, expect } from '@playwright/test';
import { TestHelpers } from './helpers/test-helpers';

test.describe('Blur Filter Tests', () => {
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
  });

  test('Test Blur 1.1: Enable Blur Filter', async ({ page }) => {
    await helpers.enableFilter('blur');
    await page.waitForTimeout(1000);
    
    const isEnabled = await helpers.isFilterEnabled('blur');
    expect(isEnabled).toBe(true);
    
    const kernelSize = await helpers.getBlurParameter('blur', 'kernel_size');
    expect(kernelSize).toBeTruthy();
  });

  test('Test Blur 1.2: Disable Blur Filter', async ({ page }) => {
    await helpers.enableFilter('blur');
    await page.waitForTimeout(1000);
    
    const wasEnabled = await helpers.isFilterEnabled('blur');
    expect(wasEnabled).toBe(true);
    
    helpers.clearConsoleLogs();
    await helpers.disableFilter('blur');
    await page.waitForTimeout(500);
    
    const isDisabled = await helpers.isFilterEnabled('blur');
    expect(isDisabled).toBe(false);
  });

  test('Test Blur 1.3: Re-enable Blur Filter', async ({ page }) => {
    await helpers.enableFilter('blur');
    await page.waitForTimeout(500);
    
    await helpers.disableFilter('blur');
    await page.waitForTimeout(500);
    
    const wasDisabled = await helpers.isFilterEnabled('blur');
    expect(wasDisabled).toBe(false);
    
    helpers.clearConsoleLogs();
    await helpers.enableFilter('blur');
    await page.waitForTimeout(500);
    
    const isEnabled = await helpers.isFilterEnabled('blur');
    expect(isEnabled).toBe(true);
  });

  test('Test Blur 2.1: Change Kernel Size', async ({ page }) => {
    await helpers.enableFilter('blur');
    await page.waitForTimeout(500);
    
    const kernelSizes = ['5', '7', '9', '11'];
    
    for (const size of kernelSizes) {
      await helpers.setBlurParameter('blur', 'kernel_size', size);
      await page.waitForTimeout(500);
      
      const currentSize = await helpers.getBlurParameter('blur', 'kernel_size');
      expect(currentSize).toBe(size);
    }
  });

  test('Test Blur 2.2: Change Sigma Value', async ({ page }) => {
    await helpers.enableFilter('blur');
    await page.waitForTimeout(500);
    
    const sigmaValues = ['1.0', '1.5', '2.0', '2.5'];
    
    for (const sigma of sigmaValues) {
      await helpers.setBlurParameter('blur', 'sigma', sigma);
      await page.waitForTimeout(500);
      
      const currentSigma = await helpers.getBlurParameter('blur', 'sigma');
      expect(parseFloat(currentSigma || '0')).toBeCloseTo(parseFloat(sigma), 1);
    }
  });

  test('Test Blur 2.3: Change Border Mode', async ({ page }) => {
    await helpers.enableFilter('blur');
    await page.waitForTimeout(500);
    
    const borderModes = ['REFLECT', 'CLAMP', 'WRAP'];
    
    for (const mode of borderModes) {
      await helpers.selectFilterParameter('blur', 'border_mode', mode);
      await page.waitForTimeout(500);
      
      const currentMode = await helpers.getSelectedParameter('blur', 'border_mode');
      expect(currentMode).toBe(mode);
    }
  });

  test('Test Blur 2.4: Toggle Separable Option', async ({ page }) => {
    await helpers.enableFilter('blur');
    await page.waitForTimeout(500);
    
    await helpers.setBlurParameter('blur', 'separable', 'false');
    await page.waitForTimeout(500);
    
    const separable = await helpers.getBlurParameter('blur', 'separable');
    expect(separable).toBe('false');
    
    await helpers.setBlurParameter('blur', 'separable', 'true');
    await page.waitForTimeout(500);
    
    const separableTrue = await helpers.getBlurParameter('blur', 'separable');
    expect(separableTrue).toBe('true');
  });

  test('Test Blur 3.1: Change Multiple Parameters', async ({ page }) => {
    await helpers.enableFilter('blur');
    await page.waitForTimeout(500);
    
    await helpers.setBlurParameter('blur', 'kernel_size', '7');
    await helpers.setBlurParameter('blur', 'sigma', '1.5');
    await helpers.selectFilterParameter('blur', 'border_mode', 'CLAMP');
    await helpers.setBlurParameter('blur', 'separable', 'false');
    await page.waitForTimeout(1000);
    
    const kernelSize = await helpers.getBlurParameter('blur', 'kernel_size');
    const sigma = await helpers.getBlurParameter('blur', 'sigma');
    const borderMode = await helpers.getSelectedParameter('blur', 'border_mode');
    const separable = await helpers.getBlurParameter('blur', 'separable');
    
    expect(kernelSize).toBe('7');
    expect(parseFloat(sigma || '0')).toBeCloseTo(1.5, 1);
    expect(borderMode).toBe('CLAMP');
    expect(separable).toBe('false');
  });

  test('Test Blur 4.1: Change Resolution While Blur Active', async ({ page }) => {
    await helpers.enableFilter('blur');
    await helpers.setBlurParameter('blur', 'kernel_size', '5');
    await helpers.setBlurParameter('blur', 'sigma', '1.0');
    
    const resolutions = ['original', 'half', 'quarter', 'original'];
    
    for (const resolution of resolutions) {
      helpers.clearConsoleLogs();
      await helpers.selectResolution(resolution);
      await page.waitForTimeout(500);
      
      const isEnabled = await helpers.isFilterEnabled('blur');
      expect(isEnabled).toBe(true);
    }
  });

  test('Test Blur 5.1: Verify Blur Filter UI Elements', async ({ page }) => {
    await helpers.enableFilter('blur');
    await page.waitForTimeout(500);
    
    const filterPanel = page.locator('filter-panel');
    await expect(filterPanel).toBeVisible();
    
    const kernelSizeControl = filterPanel.locator('[data-testid="filter-parameter-blur-kernel_size"]');
    await expect(kernelSizeControl).toBeVisible();
    
    const sigmaControl = filterPanel.locator('[data-testid="filter-parameter-blur-sigma"]');
    await expect(sigmaControl).toBeVisible();
    
    const borderModeControl = filterPanel.locator('[data-testid="filter-parameter-blur-border_mode-REFLECT"]');
    await expect(borderModeControl).toBeVisible();
    
    const separableControl = filterPanel.locator('[data-testid="filter-parameter-blur-separable"]');
    await expect(separableControl).toBeVisible();
  });

  test('Test Blur 6.1: Independent Blur Configuration Per Source', async ({ page }) => {
    for (let i = 0; i < 3; i++) {
      await helpers.addSource('lena');
    }
    
    await helpers.selectSource(1);
    await helpers.enableFilter('blur');
    await helpers.setBlurParameter('blur', 'kernel_size', '5');
    await helpers.setBlurParameter('blur', 'sigma', '1.0');
    
    await helpers.selectSource(2);
    await helpers.enableFilter('blur');
    await helpers.setBlurParameter('blur', 'kernel_size', '7');
    await helpers.setBlurParameter('blur', 'sigma', '1.5');
    
    await helpers.selectSource(1);
    await page.waitForTimeout(300);
    const kernelSize1 = await helpers.getBlurParameter('blur', 'kernel_size');
    const sigma1 = await helpers.getBlurParameter('blur', 'sigma');
    expect(kernelSize1).toBe('5');
    expect(parseFloat(sigma1 || '0')).toBeCloseTo(1.0, 1);
    
    await helpers.selectSource(2);
    await page.waitForTimeout(300);
    const kernelSize2 = await helpers.getBlurParameter('blur', 'kernel_size');
    const sigma2 = await helpers.getBlurParameter('blur', 'sigma');
    expect(kernelSize2).toBe('7');
    expect(parseFloat(sigma2 || '0')).toBeCloseTo(1.5, 1);
  });
});
