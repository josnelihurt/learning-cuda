import { test, expect } from '@playwright/test';

const STORAGE_KEY = 'cuda-app-tour-dismissed';

test.describe('First Run Tutorial', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.evaluate((key) => localStorage.removeItem(key), STORAGE_KEY);
    await page.reload();
    await page.waitForLoadState('networkidle');
  });

  test('shows tutorial on first load and can skip', async ({ page }) => {
    const overlay = page.locator('app-tour').locator('shadow=.overlay');
    await expect(overlay).toBeVisible();

    const label = page.locator('app-tour').locator('shadow=.step-label');
    await expect(label).toContainText('Step 1 of 3');

    const skipButton = page.locator('app-tour').locator('shadow=button.secondary');
    await skipButton.click();

    await expect(overlay).toBeHidden();

    await page.reload();
    await page.waitForLoadState('networkidle');
    await expect(overlay).toBeHidden();
  });

  test('returns after clearing cached state', async ({ page }) => {
    const overlay = page.locator('app-tour').locator('shadow=.overlay');
    await expect(overlay).toBeVisible();

    const nextButton = page.locator('app-tour').locator('shadow=button.primary');
    await nextButton.click();
    await nextButton.click();
    await nextButton.click();

    await expect(overlay).toBeHidden();
    await expect(page.locator('app-tour').locator('shadow=.step-label')).toHaveCount(0);

    await page.evaluate((key) => localStorage.removeItem(key), STORAGE_KEY);
    await page.reload();
    await page.waitForLoadState('networkidle');

    await expect(overlay).toBeVisible();
  });
});

