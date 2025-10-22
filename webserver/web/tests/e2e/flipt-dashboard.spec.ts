import { test, expect } from '@playwright/test';

test.describe('Flipt Dashboard Integration', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('https://localhost:8443', { waitUntil: 'networkidle' });
    });

    test('opens Flipt dashboard when clicking Flipt Feature Flags', async ({ context, page }) => {
        // Open Tools dropdown
        await page.click('[data-testid="tools-dropdown-button"]');
        await page.waitForSelector('[data-testid="tools-dropdown-menu"].open');

        // Click on Flipt Feature Flags and wait for new tab
        const [newPage] = await Promise.all([
            context.waitForEvent('page'),
            page.click('[data-testid="tool-item-flipt-feature-flags"]')
        ]);

        // Verify new tab opened with Flipt dashboard
        await newPage.waitForLoadState('networkidle');
        expect(newPage.url()).toContain('localhost:8081');
        
        // Verify Flipt UI loaded
        await expect(newPage.locator('body')).toContainText(/Flipt/i);
        
        await newPage.close();
    });
});

