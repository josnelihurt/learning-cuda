import { test, expect } from '@playwright/test';
import { createFliptApiUrl, getBaseUrl } from './utils/test-helpers';

test.describe('Flipt Flag Synchronization', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(getBaseUrl(), { waitUntil: 'networkidle' });
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
        const response = await request.get(createFliptApiUrl('/api/v1/namespaces/default/flags'));
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

    test('sync button displays success toast message', async ({ page }) => {
        // Open Tools dropdown
        await page.click('[data-testid="tools-dropdown-button"]');
        await page.waitForSelector('[data-testid="tools-dropdown-menu"].open');

        // Click Sync Feature Flags
        await page.click('[data-testid="tool-item-sync-feature-flags"]');

        // Wait for and verify success toast
        const toast = page.locator('.toast, [role="alert"], [data-testid="toast"]');
        await expect(toast).toBeVisible({ timeout: 10000 });
        await expect(toast).toContainText(/synced successfully|sync.*success/i);
    });

    test('sync button can be triggered multiple times', async ({ page }) => {
        // First sync
        await page.click('[data-testid="tools-dropdown-button"]');
        await page.click('[data-testid="tool-item-sync-feature-flags"]');
        await page.waitForTimeout(2000);

        // Second sync
        await page.click('[data-testid="tools-dropdown-button"]');
        await page.click('[data-testid="tool-item-sync-feature-flags"]');
        
        // Verify second sync also succeeds
        const toast = page.locator('.toast, [role="alert"], [data-testid="toast"]').last();
        await expect(toast).toBeVisible({ timeout: 10000 });
        await expect(toast).toContainText(/synced successfully|sync.*success/i);
    });

    test('sync button is visible and enabled in tools menu', async ({ page }) => {
        await page.click('[data-testid="tools-dropdown-button"]');
        await page.waitForSelector('[data-testid="tools-dropdown-menu"].open');

        const syncButton = page.locator('[data-testid="tool-item-sync-feature-flags"]');
        await expect(syncButton).toBeVisible();
        await expect(syncButton).toBeEnabled();
        await expect(syncButton).toContainText(/Sync Feature Flags/i);
    });
});

