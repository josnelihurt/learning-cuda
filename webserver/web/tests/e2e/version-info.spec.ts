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
    expect(count).toBeGreaterThanOrEqual(6);
    
    // Verify expected labels exist (order-independent)
    // Note: Some fields may be filtered out if empty (e.g., Proto Version)
    const requiredLabels = ['Go Version', 'C++ Version', 'Branch', 'Build Time', 'Commit Hash'];
    const optionalLabels = ['Proto Version', 'Environment'];
    const foundLabels: string[] = [];
    
    for (let i = 0; i < count; i++) {
      const item = versionItems.nth(i);
      const label = item.locator('.version-label');
      const labelText = await label.textContent();
      if (labelText) {
        foundLabels.push(labelText.trim().replace(':', ''));
      }
    }
    
    // Check that all required labels are present
    for (const expectedLabel of requiredLabels) {
      const found = foundLabels.some(label => 
        label.toLowerCase().includes(expectedLabel.toLowerCase()) || 
        expectedLabel.toLowerCase().includes(label.toLowerCase())
      );
      expect(found, `Required label "${expectedLabel}" not found. Found labels: ${foundLabels.join(', ')}`).toBeTruthy();
    }
    
    // Verify we have at least the required fields
    expect(foundLabels.length).toBeGreaterThanOrEqual(requiredLabels.length);
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
    
    const goVersion = tooltip.locator('.version-item').nth(0).locator('.version-value');
    await expect(goVersion).not.toContainText('Loading...');
    
    const cppVersion = tooltip.locator('.version-item').nth(1).locator('.version-value');
    await expect(cppVersion).not.toContainText('Loading...');
    
    const branchVersion = tooltip.locator('.version-item').filter({ hasText: 'Branch:' }).locator('.version-value');
    await expect(branchVersion).not.toContainText('Loading...');
    
    const buildVersion = tooltip.locator('.version-item').filter({ hasText: 'Build Time:' }).locator('.version-value');
    await expect(buildVersion).not.toContainText('Loading...');
    
    const commitHash = tooltip.locator('.version-item').filter({ hasText: 'Commit Hash:' }).locator('.version-value');
    await expect(commitHash).not.toContainText('Loading...');
  });

});
