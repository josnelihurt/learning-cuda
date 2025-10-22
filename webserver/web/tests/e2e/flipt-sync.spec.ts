import { test, expect } from '@playwright/test';

test.describe('Flipt Flag Synchronization', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('https://localhost:8443', { waitUntil: 'networkidle' });
    });

    test('syncs feature flags to Flipt successfully', async ({ page }) => {
        // Open Tools dropdown
        await page.click('[data-testid="tools-dropdown-button"]');
        await page.waitForSelector('[data-testid="tools-dropdown-menu"].open');

        // Click Sync Feature Flags
        await page.click('[data-testid="tool-item-sync-feature-flags"]');

        // Wait for success toast
        const toast = page.locator('toast-container');
        await expect(toast).toContainText(/synced successfully/i, { timeout: 10000 });
    });

    test('flags are created in Flipt after sync', async ({ page, request }) => {
        // Trigger sync
        await page.click('[data-testid="tools-dropdown-button"]');
        await page.click('[data-testid="tool-item-sync-feature-flags"]');
        
        // Wait for sync to complete
        await page.waitForTimeout(2000);

        // Verify flags exist in Flipt via API
        const response = await request.get('http://localhost:8081/api/v1/namespaces/default/flags');
        expect(response.status()).toBe(200);
        
        const data = await response.json();
        expect(data.flags.length).toBeGreaterThan(0);
        
        // Check specific flags
        const flagKeys = data.flags.map((f: any) => f.key);
        expect(flagKeys).toContain('ws_transport_format');
        expect(flagKeys).toContain('observability_enabled');
        expect(flagKeys).toContain('frontend_log_level');
        expect(flagKeys).toContain('frontend_console_logging');
    });

    test('displays sync button in tools menu', async ({ page }) => {
        // Open Tools dropdown
        await page.click('[data-testid="tools-dropdown-button"]');
        await page.waitForSelector('[data-testid="tools-dropdown-menu"].open');

        // Verify Sync Feature Flags option is visible
        const syncButton = page.locator('[data-testid="tool-item-sync-feature-flags"]');
        await expect(syncButton).toBeVisible();
        await expect(syncButton).toContainText(/Sync Feature Flags/i);
    });
});

