import { test, expect } from '@playwright/test';
import { TestHelpers } from './helpers/test-helpers';

test.describe('Source Removal', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
    await helpers.waitForPageReady();
    await helpers.enableDebugLogging();
  });

  test('Test 6.1: Remove Middle Source', async ({ page }) => {
    const initialCount = await helpers.getSourceCount();
    
    for (let i = 0; i < 4; i++) {
      await helpers.addSource('lena');
    }
    
    const beforeRemove = await helpers.getSourceCount();
    helpers.clearConsoleLogs();
    await helpers.removeSource(2);
    
    const count = await helpers.getSourceCount();
    expect(count).toBe(beforeRemove - 1);
    
    helpers.expectConsoleLogContains('Source removed');
    helpers.expectConsoleLogContains('Remaining:');
  });

  test('Test 6.2: Verify State Preservation After Removal', async ({ page }) => {
    const initialCount = await helpers.getSourceCount();
    
    for (let i = 0; i < 3; i++) {
      await helpers.addSource('lena');
    }
    
    await helpers.selectSource(3);
    await helpers.enableFilter('grayscale');
    await helpers.selectFilterParameter('grayscale', 'algorithm', 'average');
    
    await helpers.removeSource(2);
    
    await helpers.selectSource(3);
    await page.waitForTimeout(300);
    
    const isEnabled = await helpers.isFilterEnabled('grayscale');
    expect(isEnabled).toBe(true);
    
    const algorithm = await helpers.getSelectedParameter('grayscale', 'algorithm');
    expect(algorithm).toBe('average');
  });

});

