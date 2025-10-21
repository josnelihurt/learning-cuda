import { test, expect } from '@playwright/test';
import { TestHelpers } from './helpers/test-helpers';

test.describe('Drawer Functionality', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await helpers.enableDebugLogging();
  });

  test('Test 7.1: Open Drawer', async ({ page }) => {
    await helpers.openDrawer();
    
    const isOpen = await helpers.isDrawerOpen();
    expect(isOpen).toBe(true);
    
    const drawer = page.locator('[data-testid="source-drawer"]');
    await expect(drawer).toBeVisible();
    
    const lenaSource = page.locator('[data-testid="source-item-lena"]');
    await expect(lenaSource).toBeVisible();
  });

  test('Test 7.2: Select Source from Drawer', async ({ page }) => {
    const initialCount = await helpers.getSourceCount();
    helpers.clearConsoleLogs();
    
    await page.click('[data-testid="add-input-fab"]');
    await page.waitForSelector('[data-testid="source-drawer"]', { state: 'visible' });
    await page.click('[data-testid="source-item-lena"]');
    
    await page.waitForTimeout(500);
    
    const count = await helpers.getSourceCount();
    expect(count).toBe(initialCount + 1);
    
    helpers.expectConsoleLogContains('Source selected: lena');
    helpers.expectConsoleLogContains('Source added to grid');
  });

  test('Test 7.3: Close Drawer Without Selection', async ({ page }) => {
    const initialCount = await helpers.getSourceCount();
    
    await helpers.openDrawer();
    await helpers.closeDrawer();
    
    const isOpen = await helpers.isDrawerOpen();
    expect(isOpen).toBe(false);
    
    const count = await helpers.getSourceCount();
    expect(count).toBe(initialCount);
  });

  test('Test 7.4: Verify Duplicate Sources Allowed', async ({ page }) => {
    const initialCount = await helpers.getSourceCount();
    
    for (let i = 0; i < 3; i++) {
      await helpers.addSource('lena');
    }
    
    const count = await helpers.getSourceCount();
    expect(count).toBe(initialCount + 3);
    
    for (let i = 1; i <= count; i++) {
      const card = page.locator(`[data-source-number="${i}"]`);
      await expect(card).toBeVisible();
    }
  });

  test('Test 7.5: Close Drawer by Clicking Backdrop', async ({ page }) => {
    await helpers.openDrawer();
    
    await page.click('.backdrop');
    await page.waitForTimeout(500);
    
    const isOpen = await helpers.isDrawerOpen();
    expect(isOpen).toBe(false);
  });
});

