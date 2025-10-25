import { test, expect } from '@playwright/test';
import { TestHelpers } from './helpers/test-helpers';

test.describe('Version Info Tooltip', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await helpers.enableDebugLogging();
  });

  test('Info icon is visible in navbar-credit section', async ({ page }) => {
    const versionTooltip = page.locator('version-tooltip-lit');
    await expect(versionTooltip).toBeVisible();
    
    const infoBtn = versionTooltip.locator('button.info-btn');
    await expect(infoBtn).toBeVisible();
    
    const navbarCredit = page.locator('.navbar-credit');
    await expect(navbarCredit).toBeVisible();
    
    const creditText = navbarCredit.locator('text=by josnelihurt');
    await expect(creditText).toBeVisible();
  });

  test('Click interaction shows version tooltip', async ({ page }) => {
    const versionTooltip = page.locator('version-tooltip-lit');
    const infoBtn = versionTooltip.locator('button.info-btn');
    const tooltip = versionTooltip.locator('.tooltip');
    
    await expect(tooltip).not.toBeVisible();
    
    await infoBtn.click();
    await expect(tooltip).toBeVisible();
    
    const closeBtn = versionTooltip.locator('.tooltip-close');
    await closeBtn.click();
    await expect(tooltip).not.toBeVisible();
  });

  test('Tooltip contains version information', async ({ page }) => {
    const versionTooltip = page.locator('version-tooltip-lit');
    const infoBtn = versionTooltip.locator('button.info-btn');
    const tooltip = versionTooltip.locator('.tooltip');
    
    await infoBtn.click();
    await expect(tooltip).toBeVisible();
    
    const versionInfo = tooltip.locator('.version-info');
    await expect(versionInfo).toBeVisible();
    
    const title = tooltip.locator('.tooltip-title');
    await expect(title).toContainText('Version Information');
    
    const versionItems = versionInfo.locator('.version-item');
    const count = await versionItems.count();
    expect(count).toBe(6);
    
    const labels = ['C++:', 'Go:', 'JS:', 'Branch:', 'Build:', 'Commit:'];
    for (let i = 0; i < labels.length; i++) {
      const item = versionItems.nth(i);
      const label = item.locator('.version-label');
      await expect(label).toContainText(labels[i]);
    }
  });

  test('Tooltip is visible when clicked', async ({ page }) => {
    const versionTooltip = page.locator('version-tooltip-lit');
    const infoBtn = versionTooltip.locator('button.info-btn');
    const tooltip = versionTooltip.locator('.tooltip');
    
    await infoBtn.click();
    await expect(tooltip).toBeVisible();
    
    // Verify tooltip has content
    const versionInfo = tooltip.locator('.version-info');
    await expect(versionInfo).toBeVisible();
  });

  test('Tooltip hides when clicking close button', async ({ page }) => {
    const versionTooltip = page.locator('version-tooltip-lit');
    const infoBtn = versionTooltip.locator('button.info-btn');
    const tooltip = versionTooltip.locator('.tooltip');
    
    await infoBtn.click();
    await expect(tooltip).toBeVisible();
    
    // Click the close button instead of outside
    const closeBtn = versionTooltip.locator('.tooltip-close');
    await closeBtn.click();
    await expect(tooltip).not.toBeVisible();
  });

  test('Tooltip stays visible when hovering over it', async ({ page }) => {
    const versionTooltip = page.locator('version-tooltip-lit');
    const infoBtn = versionTooltip.locator('button.info-btn');
    const tooltip = versionTooltip.locator('.tooltip');
    
    await infoBtn.click();
    await expect(tooltip).toBeVisible();
    
    await tooltip.hover();
    await expect(tooltip).toBeVisible();
  });

  test('Version data loads correctly', async ({ page }) => {
    const versionTooltip = page.locator('version-tooltip-lit');
    const infoBtn = versionTooltip.locator('button.info-btn');
    const tooltip = versionTooltip.locator('.tooltip');
    
    await infoBtn.click();
    await expect(tooltip).toBeVisible();
    
    await page.waitForTimeout(1000);
    
    const jsVersion = tooltip.locator('.version-item').nth(2).locator('.version-value');
    await expect(jsVersion).not.toContainText('Loading...');
    
    const branchVersion = tooltip.locator('.version-item').nth(3).locator('.version-value');
    await expect(branchVersion).not.toContainText('Loading...');
    
    const buildVersion = tooltip.locator('.version-item').nth(4).locator('.version-value');
    await expect(buildVersion).not.toContainText('Loading...');
    
    const commitHash = tooltip.locator('.version-item').nth(5).locator('.version-value');
    await expect(commitHash).not.toContainText('Loading...');
  });

});
