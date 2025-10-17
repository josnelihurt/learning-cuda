import { test, expect } from '@playwright/test';
import { TestHelpers } from './helpers/test-helpers';

test.describe('Independent Filter Configuration', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    const currentCount = await helpers.getSourceCount();
    const needed = 4 - currentCount;
    
    for (let i = 0; i < needed; i++) {
      await helpers.addSource('lena');
    }
  });

  test('Test 3.1: Configure Source 1', async ({ page }) => {
    await helpers.selectSource(1);
    await helpers.enableFilter('grayscale');
    await helpers.selectFilterParameter('grayscale', 'algorithm', 'bt601');
    
    await page.waitForTimeout(1000);
    
    helpers.expectConsoleLogContains('Applying filter to source 1');
    helpers.expectConsoleLogContains('bt601');
  });

  test('Test 3.2: Configure Source 2 (Different Settings)', async ({ page }) => {
    await helpers.selectSource(2);
    
    const isEnabled = await helpers.isFilterEnabled('grayscale');
    expect(isEnabled).toBe(false);
    
    await helpers.enableFilter('grayscale');
    await helpers.selectFilterParameter('grayscale', 'algorithm', 'bt709');
    
    await page.waitForTimeout(1000);
    
    helpers.expectConsoleLogContains('Applying filter to source 2');
    helpers.expectConsoleLogContains('bt709');
  });

  test('Test 3.3: Configure Source 3 (Different Algorithm)', async ({ page }) => {
    await helpers.selectSource(3);
    await helpers.enableFilter('grayscale');
    await helpers.selectFilterParameter('grayscale', 'algorithm', 'average');
    
    await page.waitForTimeout(1000);
    
    helpers.expectConsoleLogContains('Applying filter to source 3');
    helpers.expectConsoleLogContains('average');
  });

  test('Test 3.4: Configure Source 4 (Different Algorithm)', async ({ page }) => {
    await helpers.selectSource(4);
    await helpers.enableFilter('grayscale');
    await helpers.selectFilterParameter('grayscale', 'algorithm', 'lightness');
    
    await page.waitForTimeout(1000);
    
    helpers.expectConsoleLogContains('Applying filter to source 4');
    helpers.expectConsoleLogContains('lightness');
  });

  test('Test 3.5: Verify Independent State', async ({ page }) => {
    await helpers.selectSource(1);
    await helpers.enableFilter('grayscale');
    await helpers.selectFilterParameter('grayscale', 'algorithm', 'bt601');
    
    await helpers.selectSource(2);
    await helpers.enableFilter('grayscale');
    await helpers.selectFilterParameter('grayscale', 'algorithm', 'bt709');
    
    await helpers.selectSource(1);
    const algo1 = await helpers.getSelectedParameter('grayscale', 'algorithm');
    expect(algo1).toBe('bt601');
    
    await helpers.selectSource(2);
    const algo2 = await helpers.getSelectedParameter('grayscale', 'algorithm');
    expect(algo2).toBe('bt709');
  });
});

