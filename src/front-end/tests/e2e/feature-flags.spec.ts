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

  test('should display feature flags table', async ({ page }) => {
    const featureFlagsButton = page.locator('feature-flags-button');
    const modal = page.locator('feature-flags-modal');

    await featureFlagsButton.locator('button').click();
    await expect(modal).toBeVisible();

    const table = modal.locator('.table');
    await expect(table).toBeVisible();
    await expect(modal.locator('th')).toContainText(['Key', 'Type', 'Enabled', 'Default', 'Actions']);
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

  test('should display save action for listed flags', async ({ page }) => {
    const featureFlagsButton = page.locator('feature-flags-button');
    const modal = page.locator('feature-flags-modal');

    await featureFlagsButton.locator('button').click();
    await expect(modal).toBeVisible();

    const saveButton = modal.getByRole('button', { name: 'Save' }).first();
    await expect(saveButton).toBeVisible();
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
    
    const closeBtn = headerActions.locator('.close-btn');
    await expect(closeBtn).toBeVisible();
    
    // Check modal content
    const content = modalElement.locator('.modal-content');
    await expect(content).toBeVisible();
    
    const table = content.locator('.table');
    await expect(table).toBeVisible();
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

  test('should keep modal open while editing values', async ({ page }) => {
    const featureFlagsButton = page.locator('feature-flags-button');
    const modal = page.locator('feature-flags-modal');

    await featureFlagsButton.locator('button').click();
    await expect(modal).toBeVisible();

    const defaultInput = modal.locator('tbody input[type="text"], tbody input:not([type])').first();
    if (await defaultInput.count()) {
      await defaultInput.fill('true');
    }

    await expect(modal).toBeVisible();
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
