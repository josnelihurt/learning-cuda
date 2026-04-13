import { expect, test } from '@playwright/test';

test.describe('dual-frontend-routes', () => {
  test('serves React shell at /react', async ({ page }) => {
    await page.goto('/react');
    await expect(page.locator('#root')).toBeVisible();
    await expect(page.getByText('React app loaded')).toBeVisible();
  });

  test('serves Lit shell at /lit', async ({ page }) => {
    await page.goto('/lit');
    await expect(page.locator('app-root')).toBeVisible();
  });
});
