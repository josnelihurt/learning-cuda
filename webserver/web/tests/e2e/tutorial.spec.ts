import { test, expect } from '@playwright/test';
import { TestHelpers } from './helpers/test-helpers';

const STORAGE_KEY = 'cuda-app-tour-dismissed';

test.describe('First Run Tutorial', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.addInitScript(() => {
      (window as typeof window & { __ENABLE_TOUR__?: boolean }).__ENABLE_TOUR__ = true;
    });
    await page.goto('/');
    await page.evaluate((key) => localStorage.removeItem(key), STORAGE_KEY);
    await page.reload();
    await page.waitForLoadState('domcontentloaded');
    await page.waitForSelector('[data-testid="video-grid"]', { timeout: 30000 });
    await page.waitForSelector('app-tour', { timeout: 5000, state: 'attached' });
    
    // Wait for tour to become active/visible
    await page.waitForFunction(() => {
      const tour = document.querySelector('app-tour');
      if (!tour) return false;
      const overlay = tour.shadowRoot?.querySelector('.overlay');
      return !!overlay && !overlay.classList.contains('hidden');
    }, { timeout: 10000 });
    
    await page.waitForTimeout(500);
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
    await helpers.waitForPageReady();

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
    }, { timeout: 10000 });

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
        if (next) {
          next.click();
          // Wait a bit between clicks
          const start = Date.now();
          while (Date.now() - start < 200) {}
        } else {
          break;
        }
      }
    });

    await page.waitForFunction(() => {
      const tour = document.querySelector('app-tour');
      const overlay = tour?.shadowRoot?.querySelector('.overlay');
      return !overlay || overlay.classList.contains('hidden');
    }, { timeout: 5000 });

    await page.evaluate((key) => localStorage.removeItem(key), STORAGE_KEY);
    await page.reload();
    await page.waitForLoadState('domcontentloaded');
    await page.waitForSelector('[data-testid="video-grid"]', { timeout: 30000 });
    await page.waitForSelector('app-tour', { timeout: 5000, state: 'attached' });
    
    // Wait for tour to become active/visible after reload
    await page.waitForFunction(() => {
      const tour = document.querySelector('app-tour');
      if (!tour) return false;
      const overlay = tour.shadowRoot?.querySelector('.overlay');
      return !!overlay && !overlay.classList.contains('hidden');
    }, { timeout: 10000 });
    
    await page.waitForTimeout(500);
  });
});

