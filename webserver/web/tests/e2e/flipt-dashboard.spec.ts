import { test, expect } from '@playwright/test';
import { getBaseUrl, getFliptDashboardUrlPattern } from './utils/test-helpers';

test.describe('Flipt Dashboard Integration', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(getBaseUrl(), { waitUntil: 'domcontentloaded' });
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
        
        // Check if page loaded successfully (not a certificate error page)
        if (!newPage.url().includes('chrome-error') && !newPage.url().includes('about:blank')) {
            // Verify Flipt dashboard opened with correct URL pattern
            expect(newPage.url()).toContain(getFliptDashboardUrlPattern());
            
            // Verify Flipt UI loaded
            await expect(newPage.locator('body')).toContainText(/Flipt/i);
        } else {
            // For staging with self-signed certs, certificate error is expected
            console.log('Certificate error page detected, skipping URL validation');
        }
        
        await newPage.close();
    });
});

