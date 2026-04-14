import { test, expect } from '@playwright/test';
import { TestHelpers } from './helpers/test-helpers';

test.describe('Resolution Control', () => {
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
    helpers.clearConsoleLogs();
  });

  test('Test 1.1: Original Resolution', async ({ page }) => {
    await helpers.enableFilter('grayscale');
    await helpers.selectFilterParameter('grayscale', 'algorithm', 'bt601');
    await helpers.selectResolution('original');
    
    await page.waitForTimeout(1000);
    
    helpers.expectConsoleLogContains('Sending image');
    helpers.expectConsoleLogContains('512 x 512');
  });

  test('Test 1.2: Half Resolution', async ({ page }) => {
    await helpers.enableFilter('grayscale');
    await helpers.selectFilterParameter('grayscale', 'algorithm', 'bt601');
    
    helpers.clearConsoleLogs();
    
    await helpers.selectResolution('half');
    
    await page.waitForTimeout(1000);
    
    helpers.expectConsoleLogContains('256 x 256');
  });

  test('Test 1.3: Quarter Resolution', async ({ page }) => {
    await helpers.enableFilter('grayscale');
    await helpers.selectFilterParameter('grayscale', 'algorithm', 'bt601');
    
    helpers.clearConsoleLogs();
    
    await helpers.selectResolution('quarter');
    
    await page.waitForTimeout(1000);
    
    helpers.expectConsoleLogContains('128 x 128');
  });

  test('Test 1.4: Resolution Persistence', async ({ page }) => {
    await helpers.enableFilter('grayscale');
    await helpers.selectFilterParameter('grayscale', 'algorithm', 'bt601');
    await helpers.selectResolution('half');
    
    await helpers.addSource('lena');
    
    await helpers.selectSource(1);
    await page.waitForTimeout(300);
    
    const resolution = await page.locator('[data-testid="resolution-select"]').inputValue();
    expect(resolution).toBe('half');
  });
});

