import { test, expect } from '@playwright/test';
import { TestHelpers } from './helpers/test-helpers';

test.describe('Multi-Source Management', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('Test 2.1: Add Multiple Sources', async ({ page }) => {
    const initialCount = await helpers.getSourceCount();
    
    for (let i = 1; i <= 4; i++) {
      await helpers.addSource('lena');
      const count = await helpers.getSourceCount();
      expect(count).toBe(initialCount + i);
    }
    
    const columns = await helpers.getGridColumns();
    expect(columns).toBeGreaterThanOrEqual(2);
    
    helpers.expectConsoleLogContains('Total:');
  });

  test('Test 2.2: Verify Grid Layout', async ({ page }) => {
    const initialCount = await helpers.getSourceCount();
    
    for (let i = 0; i < 4; i++) {
      await helpers.addSource('lena');
    }
    
    const cards = page.locator('[data-testid="video-source-card"]');
    const finalCount = await cards.count();
    expect(finalCount).toBe(initialCount + 4);
    
    for (let i = 1; i <= finalCount; i++) {
      const card = page.locator(`[data-source-number="${i}"]`);
      await expect(card).toBeVisible();
    }
  });

  test('Test 2.3: Maximum Capacity', async ({ page }) => {
    const initialCount = await helpers.getSourceCount();
    const toAdd = 9 - initialCount;
    
    for (let i = 0; i < toAdd; i++) {
      await helpers.addSource('lena');
    }
    
    const count = await helpers.getSourceCount();
    expect(count).toBe(9);
    
    await page.click('[data-testid="add-input-fab"]');
    await page.waitForTimeout(500);
    
    const finalCount = await helpers.getSourceCount();
    expect(finalCount).toBe(9);
  });

  test('Test 2.4: Grid Layout Transitions', async ({ page }) => {
    const initialCount = await helpers.getSourceCount();
    const targetCount = 9;
    
    for (let i = initialCount; i < targetCount; i++) {
      await helpers.addSource('lena');
      const count = await helpers.getSourceCount();
      const columns = await helpers.getGridColumns();
      
      if (count <= 2) {
        expect(columns).toBeLessThanOrEqual(1);
      } else if (count <= 4) {
        expect(columns).toBeLessThanOrEqual(2);
      } else {
        expect(columns).toBeLessThanOrEqual(3);
      }
    }
    
    const finalCount = await helpers.getSourceCount();
    expect(finalCount).toBe(targetCount);
  });
});

