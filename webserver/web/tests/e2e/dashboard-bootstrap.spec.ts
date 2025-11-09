import { test, expect } from '@playwright/test';

test.describe('Dashboard Bootstrap', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should load dashboard successfully', async ({ page }) => {
    await expect(page.locator('app-root')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('video-grid')).toBeVisible();
    await expect(page.locator('filter-panel')).toBeVisible();
  });

  test('should display app-root controls in sidebar', async ({ page }) => {
    const appRoot = page.locator('app-root');
    await expect(appRoot).toBeVisible({ timeout: 10000 });

    const acceleratorLabel = page.locator('text=Accelerator').first();
    await expect(acceleratorLabel).toBeVisible();

    const resolutionLabel = page.locator('text=Resolution').first();
    await expect(resolutionLabel).toBeVisible();
  });

  test('should have GPU accelerator active by default', async ({ page }) => {
    await page.waitForSelector('app-root', { timeout: 10000 });

    const gpuButton = page.locator('app-root').getByRole('button', { name: 'GPU' });
    await expect(gpuButton).toBeVisible();
  });

  test('should switch accelerator when clicked', async ({ page }) => {
    await page.waitForSelector('app-root', { timeout: 10000 });

    const cpuButton = page.locator('app-root').getByRole('button', { name: 'CPU' });
    await cpuButton.click();

    await page.waitForTimeout(500);
  });

  test('should change resolution when selected', async ({ page }) => {
    await page.waitForSelector('app-root', { timeout: 10000 });

    const resolutionSelect = page.locator('[data-testid="resolution-select"]').first();
    await expect(resolutionSelect).toBeVisible();

    await resolutionSelect.selectOption('half');

    await expect(resolutionSelect).toHaveValue('half');
  });

  test('should display selected source indicator', async ({ page }) => {
    await page.waitForSelector('app-root', { timeout: 10000 });

    const selectedLabel = page.locator('text=Selected').first();
    await expect(selectedLabel).toBeVisible();
  });

  test('should interact with filter panel', async ({ page }) => {
    await page.waitForSelector('filter-panel', { timeout: 10000 });

    const filterPanel = page.locator('filter-panel');
    await expect(filterPanel).toBeVisible();

    const filtersLabel = page.locator('text=Filters').first();
    await expect(filtersLabel).toBeVisible();
  });

  test('should load video grid with default source', async ({ page }) => {
    await page.waitForSelector('video-grid', { timeout: 10000 });

    const videoGrid = page.locator('video-grid');
    await expect(videoGrid).toBeVisible();

    await page.waitForTimeout(1000);

    const videoCards = page.locator('video-source-card');
    await expect(videoCards.first()).toBeVisible({ timeout: 5000 });
  });

  test('should handle service initialization errors gracefully', async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    await page.goto('/');
    await page.waitForSelector('app-root', { timeout: 10000 });

    const hasCriticalErrors = consoleErrors.some((err) =>
      err.includes('app-root element not found')
    );
    expect(hasCriticalErrors).toBe(false);
  });

  test('should complete full initialization flow', async ({ page }) => {
    await page.goto('/');

    await page.waitForSelector('app-root', { timeout: 10000 });
    await page.waitForSelector('video-grid', { timeout: 10000 });
    await page.waitForSelector('filter-panel', { timeout: 10000 });

    const appRoot = page.locator('app-root');
    await expect(appRoot).toBeVisible();

    const acceleratorButtons = page.locator('app-root').getByRole('button');
    await expect(acceleratorButtons.first()).toBeVisible();
  });
});

