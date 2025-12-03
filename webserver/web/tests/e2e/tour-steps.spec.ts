import { test, expect } from '@playwright/test';
import { TestHelpers } from './helpers/test-helpers';

const STORAGE_KEY = 'cuda-app-tour-dismissed';
const COMMIT_HASH_KEY = 'cuda-app-tour-commit-hash';

test.describe('Tour Steps', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.addInitScript(() => {
      (window as typeof window & { __ENABLE_TOUR__?: boolean }).__ENABLE_TOUR__ = true;
    });
    await page.goto('/');
    await page.evaluate((key) => localStorage.removeItem(key), STORAGE_KEY);
    await page.evaluate((key) => localStorage.removeItem(key), COMMIT_HASH_KEY);
    await page.reload();
    await page.waitForLoadState('domcontentloaded');
    await page.waitForSelector('[data-testid="video-grid"]', { timeout: 30000 });
    await page.waitForSelector('app-tour', { timeout: 5000 });
    await page.waitForTimeout(1000);
  });

  test('should show all 8 tour steps in sequence', async ({ page }) => {
    await page.waitForFunction(() => {
      const tour = document.querySelector('app-tour');
      const overlay = tour?.shadowRoot?.querySelector('.overlay');
      return !!overlay && !overlay.classList.contains('hidden');
    }, { timeout: 5000 });

    const stepTitles: string[] = [];
    const stepDescriptions: string[] = [];

    for (let i = 0; i < 8; i++) {
      const stepInfo = await page.evaluate((stepIndex) => {
        const tour = document.querySelector('app-tour');
        if (!tour) return null;
        const title = tour.shadowRoot?.querySelector('.step-title')?.textContent ?? '';
        const desc = tour.shadowRoot?.querySelector('.step-desc')?.textContent ?? '';
        const label = tour.shadowRoot?.querySelector('.step-label')?.textContent ?? '';
        return { title, desc, label, stepIndex };
      }, i);

      expect(stepInfo).not.toBeNull();
      expect(stepInfo?.label).toContain(`Step ${i + 1} of 8`);
      stepTitles.push(stepInfo!.title);
      stepDescriptions.push(stepInfo!.desc);

      if (i < 7) {
        await page.evaluate(() => {
          const tour = document.querySelector('app-tour');
          const nextButton = tour?.shadowRoot?.querySelector('button.primary') as HTMLButtonElement | undefined;
          nextButton?.click();
        });
        await page.waitForTimeout(500);
      }
    }

    expect(stepTitles).toHaveLength(8);
    expect(stepTitles[0]).toContain('Add Input');
    expect(stepTitles[1]).toContain('Filter Panel');
    expect(stepTitles[2]).toContain('Switch Images');
    expect(stepTitles[3]).toContain('Tools Menu');
    expect(stepTitles[4]).toContain('Feature Flags');
    expect(stepTitles[5]).toContain('Version Details');
    expect(stepTitles[6]).toContain('Connection Status');
    expect(stepTitles[7]).toContain('Stats Panel Toggle');
  });

  test('should navigate through all steps with Next button', async ({ page }) => {
    await page.waitForFunction(() => {
      const tour = document.querySelector('app-tour');
      const overlay = tour?.shadowRoot?.querySelector('.overlay');
      return !!overlay && !overlay.classList.contains('hidden');
    }, { timeout: 10000 });

    for (let i = 0; i < 8; i++) {
      await page.waitForTimeout(300);
      
      const currentStep = await page.evaluate((stepNum) => {
        const tour = document.querySelector('app-tour');
        if (!tour) return null;
        const label = tour.shadowRoot?.querySelector('.step-label')?.textContent ?? '';
        const title = tour.shadowRoot?.querySelector('.step-title')?.textContent ?? '';
        const overlay = tour.shadowRoot?.querySelector('.overlay');
        const isVisible = overlay && !overlay.classList.contains('hidden');
        return { label, title, isVisible, stepNum };
      }, i);

      expect(currentStep).not.toBeNull();
      expect(currentStep?.isVisible).toBe(true);
      expect(currentStep?.label).toContain(`Step ${i + 1} of 8`);

      if (i < 7) {
        const buttonText = await page.evaluate(() => {
          const tour = document.querySelector('app-tour');
          const nextButton = tour?.shadowRoot?.querySelector('button.primary') as HTMLButtonElement | undefined;
          return nextButton?.textContent?.trim() ?? '';
        });
        expect(buttonText).toBe('Next');

        await page.evaluate(() => {
          const tour = document.querySelector('app-tour');
          const nextButton = tour?.shadowRoot?.querySelector('button.primary') as HTMLButtonElement | undefined;
          nextButton?.click();
        });
        await page.waitForTimeout(800);
      } else {
        const buttonText = await page.evaluate(() => {
          const tour = document.querySelector('app-tour');
          const nextButton = tour?.shadowRoot?.querySelector('button.primary') as HTMLButtonElement | undefined;
          return nextButton?.textContent?.trim() ?? '';
        });
        expect(buttonText).toBe('Got it');
      }
    }
  });

  test('should show connection status step (step 7)', async ({ page }) => {
    await page.waitForFunction(() => {
      const tour = document.querySelector('app-tour');
      const overlay = tour?.shadowRoot?.querySelector('.overlay');
      return !!overlay && !overlay.classList.contains('hidden');
    }, { timeout: 5000 });

    for (let i = 0; i < 6; i++) {
      await page.evaluate(() => {
        const tour = document.querySelector('app-tour');
        const nextButton = tour?.shadowRoot?.querySelector('button.primary') as HTMLButtonElement | undefined;
        nextButton?.click();
      });
      await page.waitForTimeout(500);
    }

    const step7Info = await page.evaluate(() => {
      const tour = document.querySelector('app-tour');
      if (!tour) return null;
      const title = tour.shadowRoot?.querySelector('.step-title')?.textContent ?? '';
      const desc = tour.shadowRoot?.querySelector('.step-desc')?.textContent ?? '';
      const label = tour.shadowRoot?.querySelector('.step-label')?.textContent ?? '';
      return { title, desc, label };
    });

    expect(step7Info?.label).toContain('Step 7 of 8');
    expect(step7Info?.title).toContain('Connection Status');
    expect(step7Info?.desc).toContain('WebSocket');
    expect(step7Info?.desc).toContain('gRPC');
    expect(step7Info?.desc).toContain('WebRTC');
  });

  test('should show stats panel toggle step (step 8)', async ({ page }) => {
    await page.waitForFunction(() => {
      const tour = document.querySelector('app-tour');
      const overlay = tour?.shadowRoot?.querySelector('.overlay');
      return !!overlay && !overlay.classList.contains('hidden');
    }, { timeout: 5000 });

    for (let i = 0; i < 7; i++) {
      await page.evaluate(() => {
        const tour = document.querySelector('app-tour');
        const nextButton = tour?.shadowRoot?.querySelector('button.primary') as HTMLButtonElement | undefined;
        nextButton?.click();
      });
      await page.waitForTimeout(500);
    }

    const step8Info = await page.evaluate(() => {
      const tour = document.querySelector('app-tour');
      if (!tour) return null;
      const title = tour.shadowRoot?.querySelector('.step-title')?.textContent ?? '';
      const desc = tour.shadowRoot?.querySelector('.step-desc')?.textContent ?? '';
      const label = tour.shadowRoot?.querySelector('.step-label')?.textContent ?? '';
      return { title, desc, label };
    });

    expect(step8Info?.label).toContain('Step 8 of 8');
    expect(step8Info?.title).toContain('Stats Panel Toggle');
    expect(step8Info?.desc).toContain('toggle');
    expect(step8Info?.desc).toContain('collapsed');
  });

  test('should complete tour and dismiss on final step', async ({ page }) => {
    await page.waitForFunction(() => {
      const tour = document.querySelector('app-tour');
      const overlay = tour?.shadowRoot?.querySelector('.overlay');
      return !!overlay && !overlay.classList.contains('hidden');
    }, { timeout: 5000 });

    for (let i = 0; i < 8; i++) {
      await page.evaluate(() => {
        const tour = document.querySelector('app-tour');
        const nextButton = tour?.shadowRoot?.querySelector('button.primary') as HTMLButtonElement | undefined;
        nextButton?.click();
      });
      await page.waitForTimeout(500);
    }

    await page.waitForFunction(() => {
      const tour = document.querySelector('app-tour');
      const overlay = tour?.shadowRoot?.querySelector('.overlay');
      return !overlay || overlay.classList.contains('hidden');
    }, { timeout: 2000 });

    const dismissed = await page.evaluate((key) => {
      return localStorage.getItem(key) === 'true';
    }, STORAGE_KEY);

    expect(dismissed).toBe(true);
  });

  test('should allow skipping tour at any step', async ({ page }) => {
    await page.waitForFunction(() => {
      const tour = document.querySelector('app-tour');
      const overlay = tour?.shadowRoot?.querySelector('.overlay');
      return !!overlay && !overlay.classList.contains('hidden');
    }, { timeout: 5000 });

    await page.evaluate(() => {
      const tour = document.querySelector('app-tour');
      const skipButton = tour?.shadowRoot?.querySelector('button.secondary') as HTMLButtonElement | undefined;
      skipButton?.click();
    });

    await page.waitForFunction(() => {
      const tour = document.querySelector('app-tour');
      const overlay = tour?.shadowRoot?.querySelector('.overlay');
      return !overlay || overlay.classList.contains('hidden');
    }, { timeout: 2000 });

    const dismissed = await page.evaluate((key) => {
      return localStorage.getItem(key) === 'true';
    }, STORAGE_KEY);

    expect(dismissed).toBe(true);
  });
});

