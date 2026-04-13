import { test, expect } from '@playwright/test';
import { TestHelpers } from './helpers/test-helpers';

test.describe('WebSocket Management', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
    await helpers.waitForPageReady();
  });


  test('Test 8.3: WebSocket Stability', async ({ page }) => {
    for (let i = 0; i < 3; i++) {
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
    
    await page.waitForTimeout(2000);
    
    for (let i = 1; i <= 3; i++) {
      const card = page.locator(`[data-source-number="${i}"]`);
      await expect(card).toBeVisible();
    }
  });

  test('Test 8.4: Multiple Filter Changes', async ({ page }) => {
    await helpers.addSource('lena');
    await helpers.selectSource(1);
    
    await helpers.enableFilter('grayscale');
    
    const algorithms = ['bt601', 'bt709', 'average', 'lightness'];
    for (const algo of algorithms) {
      await helpers.selectFilterParameter('grayscale', 'algorithm', algo);
      await page.waitForTimeout(500);
    }
    
    const card = page.locator('[data-source-number="1"]');
    await expect(card).toBeVisible();
  });
});

