import { test, expect } from '@playwright/test';
import { TestHelpers } from './helpers/test-helpers';

test.describe('Panel Synchronization', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await helpers.enableDebugLogging();
    
    const currentCount = await helpers.getSourceCount();
    const needed = 4 - currentCount;
    
    for (let i = 0; i < needed; i++) {
      await helpers.addSource('lena');
    }
    
    await helpers.selectSource(1);
    await helpers.enableFilter('grayscale');
    await helpers.selectFilterParameter('grayscale', 'algorithm', 'bt601');
    
    await helpers.selectSource(2);
    await helpers.enableFilter('grayscale');
    await helpers.selectFilterParameter('grayscale', 'algorithm', 'bt709');
    
    await helpers.selectSource(3);
    await helpers.enableFilter('grayscale');
    await helpers.selectFilterParameter('grayscale', 'algorithm', 'average');
    
    await helpers.selectSource(4);
    await helpers.enableFilter('grayscale');
    await helpers.selectFilterParameter('grayscale', 'algorithm', 'lightness');
  });

  test('Test 4.1: Verify Source 1 Panel State', async ({ page }) => {
    await helpers.selectSource(1);
    await page.waitForTimeout(200);
    
    const isChecked = await helpers.isFilterEnabled('grayscale');
    expect(isChecked).toBe(true);
    
    const algorithm = await helpers.getSelectedParameter('grayscale', 'algorithm');
    expect(algorithm).toBe('bt601');
  });

  test('Test 4.2: Verify Source 2 Panel State', async ({ page }) => {
    await helpers.selectSource(2);
    await page.waitForTimeout(200);
    
    const isChecked = await helpers.isFilterEnabled('grayscale');
    expect(isChecked).toBe(true);
    
    const algorithm = await helpers.getSelectedParameter('grayscale', 'algorithm');
    expect(algorithm).toBe('bt709');
  });

  test('Test 4.3: Verify Source 3 Panel State', async ({ page }) => {
    await helpers.selectSource(3);
    await page.waitForTimeout(200);
    
    const isChecked = await helpers.isFilterEnabled('grayscale');
    expect(isChecked).toBe(true);
    
    const algorithm = await helpers.getSelectedParameter('grayscale', 'algorithm');
    expect(algorithm).toBe('average');
  });

  test('Test 4.4: Verify Source 4 Panel State', async ({ page }) => {
    await helpers.selectSource(4);
    await page.waitForTimeout(200);
    
    const isChecked = await helpers.isFilterEnabled('grayscale');
    expect(isChecked).toBe(true);
    
    const algorithm = await helpers.getSelectedParameter('grayscale', 'algorithm');
    expect(algorithm).toBe('lightness');
  });

  test('Test 4.5: Rapid Source Switching', async ({ page }) => {
    test.setTimeout(30000);
    
    for (let cycle = 0; cycle < 2; cycle++) {
      for (let source = 1; source <= 4; source++) {
        await helpers.selectSource(source);
        await page.waitForTimeout(200);
        
        const isChecked = await helpers.isFilterEnabled('grayscale');
        expect(isChecked).toBe(true);
        
        const algorithm = await helpers.getSelectedParameter('grayscale', 'algorithm');
        expect(algorithm).toBeTruthy();
      }
    }
    
    const foundLog = await helpers.waitForConsoleLog('Card selected', 2000);
    expect(foundLog).toBe(true);
  });
});

