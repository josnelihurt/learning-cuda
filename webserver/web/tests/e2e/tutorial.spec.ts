import { test, expect } from '@playwright/test';

const STORAGE_KEY = 'cuda-app-tour-dismissed';

test.describe('First Run Tutorial', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      (window as typeof window & { __ENABLE_TOUR__?: boolean }).__ENABLE_TOUR__ = true;
    });
    await page.goto('/');
    await page.evaluate((key) => localStorage.removeItem(key), STORAGE_KEY);
    await page.reload();
    await page.waitForLoadState('networkidle');
  });

  test('shows tutorial on first load and can skip', async ({ page }) => {
    await page.waitForFunction(() => {
      const tour = document.querySelector('app-tour');
      const overlay = tour?.shadowRoot?.querySelector('.overlay');
      return !!overlay && !overlay.classList.contains('hidden');
    });

    const labelText = await page.evaluate(() => {
      const tour = document.querySelector('app-tour');
      const label = tour?.shadowRoot?.querySelector('.step-label');
      return label?.textContent ?? '';
    });
    expect(labelText).toContain('Step 1 of');

    await page.evaluate(() => {
      const tour = document.querySelector('app-tour');
      const skip = tour?.shadowRoot?.querySelector('button.secondary') as HTMLButtonElement | undefined;
      skip?.click();
    });

    await page.waitForFunction(() => {
      const tour = document.querySelector('app-tour');
      const overlay = tour?.shadowRoot?.querySelector('.overlay');
      return !overlay || overlay.classList.contains('hidden');
    });

    await page.reload();
    await page.waitForLoadState('networkidle');

    const overlayHiddenAfterReload = await page.evaluate(() => {
      const tour = document.querySelector('app-tour');
      const overlay = tour?.shadowRoot?.querySelector('.overlay');
      return !overlay || overlay.classList.contains('hidden');
    });
    expect(overlayHiddenAfterReload).toBe(true);
  });

  test('returns after clearing cached state', async ({ page }) => {
    await page.waitForFunction(() => {
      const tour = document.querySelector('app-tour');
      const overlay = tour?.shadowRoot?.querySelector('.overlay');
      return !!overlay && !overlay.classList.contains('hidden');
    });

    await page.evaluate(() => {
      const tour = document.querySelector('app-tour');
      if (!tour) return;
      let safety = 10;
      while (safety-- > 0) {
        const overlay = tour.shadowRoot?.querySelector('.overlay');
        if (!overlay || overlay.classList.contains('hidden')) {
          break;
        }
        const next = tour.shadowRoot?.querySelector('button.primary') as HTMLButtonElement | undefined;
        next?.click();
      }
    });

    await page.waitForFunction(() => {
      const tour = document.querySelector('app-tour');
      const overlay = tour?.shadowRoot?.querySelector('.overlay');
      return !overlay || overlay.classList.contains('hidden');
    });

    await page.evaluate((key) => localStorage.removeItem(key), STORAGE_KEY);
    await page.reload();
    await page.waitForLoadState('networkidle');

    await page.waitForFunction(() => {
      const tour = document.querySelector('app-tour');
      const overlay = tour?.shadowRoot?.querySelector('.overlay');
      return !!overlay && !overlay.classList.contains('hidden');
    });
  });
});

