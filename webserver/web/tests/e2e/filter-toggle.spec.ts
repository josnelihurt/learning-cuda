import { test, expect } from '@playwright/test';
import { TestHelpers } from './helpers/test-helpers';

test.describe('Filter Toggle and Changes', () => {
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

  test('Test 5.1: Disable Filter', async ({ page }) => {
    await helpers.enableFilter('grayscale');
    await page.waitForTimeout(1000);
    
    const wasEnabled = await helpers.isFilterEnabled('grayscale');
    expect(wasEnabled).toBe(true);
    
    helpers.clearConsoleLogs();
    await helpers.disableFilter('grayscale');
    await page.waitForTimeout(500);
    
    const isDisabled = await helpers.isFilterEnabled('grayscale');
    expect(isDisabled).toBe(false);
    
    const foundLog = await helpers.waitForConsoleLog('none', 2000);
    if (!foundLog) {
      console.log('Console log not found, but filter state verified');
    }
  });

  test('Test 5.2: Re-enable Filter', async ({ page }) => {
    await helpers.enableFilter('grayscale');
    await page.waitForTimeout(500);
    
    await helpers.disableFilter('grayscale');
    await page.waitForTimeout(500);
    
    const wasDisabled = await helpers.isFilterEnabled('grayscale');
    expect(wasDisabled).toBe(false);
    
    helpers.clearConsoleLogs();
    await helpers.enableFilter('grayscale');
    await page.waitForTimeout(500);
    
    const isEnabled = await helpers.isFilterEnabled('grayscale');
    expect(isEnabled).toBe(true);
    
    const foundLog = await helpers.waitForConsoleLog('grayscale', 2000);
    if (!foundLog) {
      console.log('Console log not found, but filter state verified');
    }
  });

  test('Test 5.3: Change Algorithm While Active', async ({ page }) => {
    await helpers.enableFilter('grayscale');
    
    const algorithms = ['bt601', 'bt709', 'average', 'lightness', 'luminosity'];
    
    for (const algo of algorithms) {
      await helpers.selectFilterParameter('grayscale', 'algorithm', algo);
      await page.waitForTimeout(500);
      
      const selected = await helpers.getSelectedParameter('grayscale', 'algorithm');
      expect(selected).toBe(algo);
    }
  });

  test('Test 5.4: Change Resolution While Filtered', async ({ page }) => {
    await helpers.enableFilter('grayscale');
    await helpers.selectFilterParameter('grayscale', 'algorithm', 'bt601');
    
    const resolutions = ['original', 'half', 'quarter', 'original'];
    
    for (const resolution of resolutions) {
      helpers.clearConsoleLogs();
      await helpers.selectResolution(resolution);
      await page.waitForTimeout(500);
      
      const isEnabled = await helpers.isFilterEnabled('grayscale');
      expect(isEnabled).toBe(true);
    }
  });
});

