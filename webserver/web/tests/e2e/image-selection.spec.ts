import { test, expect } from '@playwright/test';
import { TestHelpers } from './helpers/test-helpers';

test.describe('Image Selection', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
    await helpers.waitForPageReady();
    await helpers.enableDebugLogging();
  });

  test('Test 8.1: Change image button visible only on static sources', async ({ page }) => {
    await helpers.addSource('lena');
    
    const card = page.locator('[data-source-number="1"]');
    const changeBtn = card.locator('[data-testid="change-image-button"]');
    await expect(changeBtn).toBeVisible();
  });

  test('Test 8.2: Open image selector modal from static source', async ({ page }) => {
    await helpers.addSource('lena');
    
    const card = page.locator('[data-source-number="1"]');
    await card.locator('[data-testid="change-image-button"]').click();
    
    const modal = page.locator('image-selector-modal[open]');
    await expect(modal).toBeVisible();
  });

  test('Test 8.3: Display available images in modal', async ({ page }) => {
    await helpers.addSource('lena');
    
    const card = page.locator('[data-source-number="1"]');
    await card.locator('[data-testid="change-image-button"]').click();
    
    const modal = page.locator('image-selector-modal[open]');
    await expect(modal).toBeVisible();
    
    const lenaItem = modal.locator('[data-testid="image-item-lena"]');
    await expect(lenaItem).toBeVisible();
    const lenaImg = lenaItem.locator('img.image-preview');
    await expect(lenaImg).toBeVisible();
    await expect(lenaImg).toHaveAttribute('src', /lena\.png/);
    
    const mandrillItem = modal.locator('[data-testid="image-item-mandrill"]');
    await expect(mandrillItem).toBeVisible();
    const mandrillImg = mandrillItem.locator('img.image-preview');
    await expect(mandrillImg).toBeVisible();
    
    const peppersItem = modal.locator('[data-testid="image-item-peppers"]');
    await expect(peppersItem).toBeVisible();
    
    const barbaraItem = modal.locator('[data-testid="image-item-barbara"]');
    await expect(barbaraItem).toBeVisible();
  });

  test('Test 8.4: Select image and verify source updates', async ({ page }) => {
    await helpers.addSource('lena');
    
    const card = page.locator('[data-source-number="1"]');
    await card.locator('[data-testid="change-image-button"]').click();
    
    const modal = page.locator('image-selector-modal');
    await expect(modal).toHaveAttribute('open');
    
    const mandrillItem = modal.locator('[data-testid="image-item-mandrill"]');
    await mandrillItem.click();
    
    await expect(modal).not.toHaveAttribute('open');
    
    helpers.expectConsoleLogContains('Image selected: mandrill');
    helpers.expectConsoleLogContains('Source image changed');
  });

  test('Test 8.5: Close modal without selection', async ({ page }) => {
    await helpers.addSource('lena');
    
    const card = page.locator('[data-source-number="1"]');
    await card.locator('[data-testid="change-image-button"]').click();
    
    const modal = page.locator('image-selector-modal');
    await expect(modal).toHaveAttribute('open');
    
    await modal.locator('[data-testid="modal-close"]').click();
    await expect(modal).not.toHaveAttribute('open');
  });

  test('Test 8.6: Close modal by clicking backdrop', async ({ page }) => {
    await helpers.addSource('lena');
    
    const card = page.locator('[data-source-number="1"]');
    await card.locator('[data-testid="change-image-button"]').click();
    
    const modal = page.locator('image-selector-modal');
    await expect(modal).toHaveAttribute('open');
    
    await modal.locator('.backdrop').click({ force: true });
    await expect(modal).not.toHaveAttribute('open');
  });

  test('Test 8.7: Change image button not visible on camera sources', async ({ page }) => {
    const initialCount = await helpers.getSourceCount();
    
    await helpers.addSource('webcam');
    
    const card = page.locator(`[data-source-number="${initialCount + 1}"]`);
    await card.waitFor({ state: 'visible' });
    
    const changeBtn = card.locator('[data-testid="change-image-button"]');
    await expect(changeBtn).not.toBeVisible();
  });

  test('Test 8.8: Multiple image changes on same source', async ({ page }) => {
    await helpers.addSource('lena');
    
    const card = page.locator('[data-source-number="1"]');
    const modal = page.locator('image-selector-modal');
    
    await card.locator('[data-testid="change-image-button"]').click();
    await expect(modal).toHaveAttribute('open');
    
    await modal.locator('[data-testid="image-item-mandrill"]').click();
    await expect(modal).not.toHaveAttribute('open');
    
    helpers.expectConsoleLogContains('Image selected: mandrill');
    
    await card.locator('[data-testid="change-image-button"]').click();
    await expect(modal).toHaveAttribute('open');
    
    await modal.locator('[data-testid="image-item-peppers"]').click();
    await expect(modal).not.toHaveAttribute('open');
    
    helpers.expectConsoleLogContains('Image selected: peppers');
  });
});

