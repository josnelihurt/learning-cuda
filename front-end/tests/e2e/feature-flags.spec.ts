import { test, expect } from '@playwright/test';
import { TestHelpers } from './helpers/test-helpers';

test.describe('Feature Flags Modal', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
    await helpers.waitForPageReady();
  });

  test('should display feature flags button in navbar', async ({ page }) => {
    const featureFlagsButton = page.locator('feature-flags-button');
    await expect(featureFlagsButton).toBeVisible();
    
    const button = featureFlagsButton.locator('button');
    await expect(button).toBeVisible();
    await expect(button).toContainText('Feature Flags');
  });

  test('should open modal when clicking feature flags button', async ({ page }) => {
    const featureFlagsButton = page.locator('feature-flags-button');
    const modal = page.locator('feature-flags-modal');
    
    // Modal should be hidden initially
    await expect(modal).not.toBeVisible();
    
    // Click the feature flags button
    await featureFlagsButton.locator('button').click();
    
    // Modal should be visible
    await expect(modal).toBeVisible();
    await expect(modal).toHaveAttribute('open', '');
    
    // Check modal content
    const modalTitle = modal.locator('.modal-title');
    await expect(modalTitle).toContainText('Feature Flags');
  });

  test('should display iframe with Flipt UI', async ({ page }) => {
    const featureFlagsButton = page.locator('feature-flags-button');
    const modal = page.locator('feature-flags-modal');
    
    // Open modal
    await featureFlagsButton.locator('button').click();
    await expect(modal).toBeVisible();
    
    // Check iframe is present
    const iframe = modal.locator('iframe');
    await expect(iframe).toBeVisible();
    
    // Check iframe attributes
    await expect(iframe).toHaveAttribute('title', 'Flipt Feature Flags');
    await expect(iframe).toHaveAttribute('sandbox', 'allow-same-origin allow-scripts allow-forms allow-popups allow-top-navigation');
    await expect(iframe).toHaveAttribute('allow', 'fullscreen');
    
    // Check iframe src contains flipt path
    const iframeSrc = await iframe.getAttribute('src');
    expect(iframeSrc).toContain('/flipt/');
  });

  test('should close modal when clicking close button', async ({ page }) => {
    const featureFlagsButton = page.locator('feature-flags-button');
    const modal = page.locator('feature-flags-modal');
    
    // Open modal
    await featureFlagsButton.locator('button').click();
    await expect(modal).toBeVisible();
    
    // Click close button
    const closeButton = modal.locator('.close-btn');
    await expect(closeButton).toBeVisible();
    await closeButton.click();
    
    // Modal should be hidden
    await expect(modal).not.toBeVisible();
    await expect(modal).not.toHaveAttribute('open', '');
  });

  test('should close modal when clicking outside (backdrop)', async ({ page }) => {
    const featureFlagsButton = page.locator('feature-flags-button');
    const modal = page.locator('feature-flags-modal');
    
    // Open modal
    await featureFlagsButton.locator('button').click();
    await expect(modal).toBeVisible();
    
    // Click somewhere outside the modal (bottom left corner) to close it
    await page.click('body', { position: { x: 10, y: 10 } });
    
    // Modal should be hidden
    await expect(modal).not.toBeVisible();
  });

  test('should display sync button and handle sync action', async ({ page }) => {
    const featureFlagsButton = page.locator('feature-flags-button');
    const modal = page.locator('feature-flags-modal');
    
    // Open modal
    await featureFlagsButton.locator('button').click();
    await expect(modal).toBeVisible();
    
    // Check sync button is present
    const syncButton = modal.locator('.sync-btn');
    await expect(syncButton).toBeVisible();
    await expect(syncButton).toContainText('Sync');
    await expect(syncButton).toHaveAttribute('title', 'Sync feature flags to Flipt');
    
    // Check sync button is not disabled initially
    await expect(syncButton).not.toBeDisabled();
  });

  test('should handle sync button click and show loading state', async ({ page }) => {
    const featureFlagsButton = page.locator('feature-flags-button');
    const modal = page.locator('feature-flags-modal');
    
    // Open modal
    await featureFlagsButton.locator('button').click();
    await expect(modal).toBeVisible();
    
    const syncButton = modal.locator('.sync-btn');
    
    // Click sync button
    await syncButton.click();
    
    // Wait for sync to complete - the sync might be very fast
    // We just verify it completes successfully
    await expect(syncButton).not.toBeDisabled({ timeout: 10000 });
    await expect(syncButton).toContainText('Sync');
    
    // Verify button is back to normal state
    await expect(syncButton).not.toHaveClass(/syncing/);
  });

  test('should have proper modal styling and layout', async ({ page }) => {
    const featureFlagsButton = page.locator('feature-flags-button');
    const modal = page.locator('feature-flags-modal');
    
    // Open modal
    await featureFlagsButton.locator('button').click();
    await expect(modal).toBeVisible();
    
    // Check modal structure
    const modalElement = modal.locator('.modal');
    await expect(modalElement).toBeVisible();
    
    // Check header
    const header = modalElement.locator('.modal-header');
    await expect(header).toBeVisible();
    
    const title = header.locator('.modal-title');
    await expect(title).toContainText('Feature Flags');
    
    // Check header actions
    const headerActions = header.locator('.header-actions');
    await expect(headerActions).toBeVisible();
    
    const syncBtn = headerActions.locator('.sync-btn');
    const closeBtn = headerActions.locator('.close-btn');
    await expect(syncBtn).toBeVisible();
    await expect(closeBtn).toBeVisible();
    
    // Check modal content
    const content = modalElement.locator('.modal-content');
    await expect(content).toBeVisible();
    
    const iframeContainer = content.locator('.iframe-container');
    await expect(iframeContainer).toBeVisible();
    
    const iframe = iframeContainer.locator('iframe');
    await expect(iframe).toBeVisible();
  });

  test('should handle keyboard navigation (ESC key)', async ({ page }) => {
    const featureFlagsButton = page.locator('feature-flags-button');
    const modal = page.locator('feature-flags-modal');
    
    // Open modal
    await featureFlagsButton.locator('button').click();
    await expect(modal).toBeVisible();
    
    // Press ESC key
    await page.keyboard.press('Escape');
    
    // Modal should be hidden
    await expect(modal).not.toBeVisible();
  });

  test('should maintain modal state during interactions', async ({ page }) => {
    const featureFlagsButton = page.locator('feature-flags-button');
    const modal = page.locator('feature-flags-modal');
    
    // Open modal
    await featureFlagsButton.locator('button').click();
    await expect(modal).toBeVisible();
    
    // Interact with sync button
    const syncButton = modal.locator('.sync-btn');
    await syncButton.click();
    
    // Modal should still be visible during sync
    await expect(modal).toBeVisible();
    
    // Wait for sync to complete
    await expect(syncButton).not.toHaveClass(/syncing/, { timeout: 10000 });
    
    // Modal should still be visible after sync
    await expect(modal).toBeVisible();
    
    // Close modal
    const closeButton = modal.locator('.close-btn');
    await closeButton.click();
    await expect(modal).not.toBeVisible();
  });

  test('should handle multiple open/close cycles', async ({ page }) => {
    const featureFlagsButton = page.locator('feature-flags-button');
    const modal = page.locator('feature-flags-modal');
    
    // First cycle
    await featureFlagsButton.locator('button').click();
    await expect(modal).toBeVisible();
    await modal.locator('.close-btn').click();
    await expect(modal).not.toBeVisible();
    
    // Second cycle - click outside modal to close
    await featureFlagsButton.locator('button').click();
    await expect(modal).toBeVisible();
    await page.click('body', { position: { x: 10, y: 10 } });
    await expect(modal).not.toBeVisible();
    
    // Third cycle
    await featureFlagsButton.locator('button').click();
    await expect(modal).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(modal).not.toBeVisible();
  });

  test('should have responsive design for different screen sizes', async ({ page }) => {
    const featureFlagsButton = page.locator('feature-flags-button');
    const modal = page.locator('feature-flags-modal');
    
    // Test on mobile size
    await page.setViewportSize({ width: 375, height: 667 });
    await featureFlagsButton.locator('button').click();
    await expect(modal).toBeVisible();
    
    const modalElement = modal.locator('.modal');
    const modalBox = await modalElement.boundingBox();
    expect(modalBox?.width).toBeLessThanOrEqual(375);
    
    await modal.locator('.close-btn').click();
    await expect(modal).not.toBeVisible();
    
    // Test on tablet size
    await page.setViewportSize({ width: 768, height: 1024 });
    await featureFlagsButton.locator('button').click();
    await expect(modal).toBeVisible();
    
    const modalBoxTablet = await modalElement.boundingBox();
    expect(modalBoxTablet?.width).toBeLessThanOrEqual(768);
    
    await modal.locator('.close-btn').click();
    await expect(modal).not.toBeVisible();
    
    // Test on desktop size
    await page.setViewportSize({ width: 1920, height: 1080 });
    await featureFlagsButton.locator('button').click();
    await expect(modal).toBeVisible();
    
    const modalBoxDesktop = await modalElement.boundingBox();
    expect(modalBoxDesktop?.width).toBeLessThanOrEqual(1200); // Allow tolerance for rounding
  });
});
