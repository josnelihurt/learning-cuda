import { test, expect } from '@playwright/test';
import { TestHelpers } from './helpers/test-helpers';

test.describe('UI Validation', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('Validate Page Title and Header', async ({ page }) => {
    await expect(page).toHaveTitle(/CUDA/i);
    
    const header = page.locator('header');
    await expect(header).toBeVisible();
  });

  test('Validate Tools Dropdown', async ({ page }) => {
    const toolsButton = page.locator('[data-testid="tools-dropdown-button"]');
    await expect(toolsButton).toBeVisible();
    await expect(toolsButton).toHaveText(/Tools/);
    
    await toolsButton.click();
    
    const toolsMenu = page.locator('[data-testid="tools-dropdown-menu"]');
    await expect(toolsMenu).toBeVisible();
    
    await page.waitForTimeout(300);
    
    const menuItems = toolsMenu.locator('.dropdown-item');
    const count = await menuItems.count();
    expect(count).toBeGreaterThan(0);
  });

  test('Validate Add Input FAB Button', async ({ page }) => {
    const fab = page.locator('[data-testid="add-input-fab"]');
    await expect(fab).toBeVisible();
    await expect(fab).toContainText(/Add Input/);
    
    await fab.hover();
    await expect(fab).toBeEnabled();
  });

  test('Validate Grid With Default Source', async ({ page }) => {
    const grid = page.locator('[data-testid="video-grid"]');
    await expect(grid).toBeVisible();
    
    const cards = page.locator('[data-testid="video-source-card"]');
    const count = await cards.count();
    expect(count).toBeGreaterThanOrEqual(1);
  });

  test('Validate Filter Panel Structure', async ({ page }) => {
    await helpers.addSource('lena');
    await helpers.selectSource(1);
    
    const filterPanel = page.locator('filter-panel');
    await expect(filterPanel).toBeVisible();
    
    const filtersSection = filterPanel.locator('.filters-section');
    await expect(filtersSection).toBeVisible();
    
    const controlLabel = filtersSection.locator('.control-label');
    await expect(controlLabel).toContainText(/Filters/);
  });

  test('Validate Filter Options', async ({ page }) => {
    await helpers.addSource('lena');
    await helpers.selectSource(1);
    
    const grayscaleCheckbox = page.locator('[data-testid="filter-checkbox-grayscale"]');
    await expect(grayscaleCheckbox).toBeVisible();
    
    await helpers.enableFilter('grayscale');
    
    const bt601Radio = page.locator('[data-testid="filter-parameter-grayscale-algorithm-bt601"]');
    await expect(bt601Radio).toBeVisible();
    
    const bt709Radio = page.locator('[data-testid="filter-parameter-grayscale-algorithm-bt709"]');
    await expect(bt709Radio).toBeVisible();
    
    const averageRadio = page.locator('[data-testid="filter-parameter-grayscale-algorithm-average"]');
    await expect(averageRadio).toBeVisible();
  });

  test('Validate Source Drawer Content', async ({ page }) => {
    await helpers.openDrawer();
    
    const drawer = page.locator('[data-testid="source-drawer"]');
    await expect(drawer).toBeVisible();
    
    const title = drawer.locator('.drawer-title');
    await expect(title).toContainText(/Select Input Source/);
    
    const lenaItem = page.locator('[data-testid="source-item-lena"]');
    await expect(lenaItem).toBeVisible();
    
    const lenaName = lenaItem.locator('.source-name');
    await expect(lenaName).toContainText(/Lena/i);
    
    const lenaType = lenaItem.locator('.source-type');
    await expect(lenaType).toContainText(/static/i);
  });

  test('Validate Video Source Card Structure', async ({ page }) => {
    const initialCount = await helpers.getSourceCount();
    if (initialCount === 0) {
      await helpers.addSource('lena');
    }
    
    const card = page.locator('[data-source-number="1"]');
    await expect(card).toBeVisible();
    
    const sourceNumber = card.locator('.source-number');
    await expect(sourceNumber).toContainText('1');
    
    const closeButton = card.locator('[data-testid="source-close-button"]');
    await expect(closeButton).toBeAttached();
  });

  test('Validate Grid Layout Classes', async ({ page }) => {
    await helpers.clearAllSources();
    
    await helpers.addSource('lena');
    const grid = page.locator('[data-testid="video-grid"]');
    await expect(grid).toHaveClass(/grid-1/);
    
    await helpers.addSource('lena');
    await expect(grid).toHaveClass(/grid-2/);
    
    await helpers.addSource('lena');
    await helpers.addSource('lena');
    await expect(grid).toHaveClass(/grid-4/);
  });

  test('Validate Responsive Elements', async ({ page }) => {
    const fab = page.locator('[data-testid="add-input-fab"]');
    const fabBox = await fab.boundingBox();
    expect(fabBox).toBeTruthy();
    expect(fabBox!.width).toBeGreaterThan(0);
    expect(fabBox!.height).toBeGreaterThan(0);
    
    const toolsDropdown = page.locator('[data-testid="tools-dropdown-button"]');
    const dropdownBox = await toolsDropdown.boundingBox();
    expect(dropdownBox).toBeTruthy();
  });

  test('Validate Console Logging', async ({ page }) => {
    await helpers.addSource('lena');
    
    await page.waitForTimeout(500);
    
    const logs = helpers.getConsoleLogs();
    expect(logs.length).toBeGreaterThan(0);
    
    const hasSourceLog = logs.some(log => log.text.includes('Source added to grid'));
    expect(hasSourceLog).toBe(true);
  });
});

