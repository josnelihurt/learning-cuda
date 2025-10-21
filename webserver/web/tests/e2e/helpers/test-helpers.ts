import { Page, expect } from '@playwright/test';

export interface ConsoleLog {
  type: string;
  text: string;
  timestamp: Date;
}

export class TestHelpers {
  private consoleLogs: ConsoleLog[] = [];

  constructor(private page: Page) {
    this.setupConsoleCapture();
  }

  private setupConsoleCapture() {
    this.page.on('console', msg => {
      this.consoleLogs.push({
        type: msg.type(),
        text: msg.text(),
        timestamp: new Date()
      });
    });
  }

  async enableDebugLogging(): Promise<void> {
    await this.page.evaluate(() => {
      const logger = (window as any).logger;
      if (logger && typeof logger.initialize === 'function') {
        logger.initialize('DEBUG', true);
      }
    });
  }

  getConsoleLogs(): ConsoleLog[] {
    return this.consoleLogs;
  }

  clearConsoleLogs(): void {
    this.consoleLogs = [];
  }

  async waitForConsoleLog(pattern: string | RegExp, timeout = 5000): Promise<boolean> {
    const startTime = Date.now();
    while (Date.now() - startTime < timeout) {
      const found = this.consoleLogs.some(log => {
        if (typeof pattern === 'string') {
          return log.text.includes(pattern);
        }
        return pattern.test(log.text);
      });
      if (found) return true;
      await this.page.waitForTimeout(100);
    }
    return false;
  }

  async addSource(sourceName: string): Promise<void> {
    await this.page.click('[data-testid="add-input-fab"]');
    await this.page.waitForSelector('[data-testid="source-drawer"]', { state: 'visible' });
    await this.page.click(`[data-testid="source-item-${sourceName}"]`);
    await this.page.waitForTimeout(300);
  }

  async selectSource(sourceNumber: number): Promise<void> {
    const card = this.page.locator(`[data-source-number="${sourceNumber}"]`);
    const sourceNumberBadge = card.locator('.source-number');
    await sourceNumberBadge.click();
    await this.page.waitForTimeout(100);
  }

  async removeSource(sourceNumber: number): Promise<void> {
    const card = this.page.locator(`[data-source-number="${sourceNumber}"]`);
    const closeButton = card.locator('[data-testid="source-close-button"]');
    await closeButton.click({ force: true });
    await this.page.waitForTimeout(300);
  }

  async clearAllSources(): Promise<void> {
    let count = await this.getSourceCount();
    while (count > 0) {
      await this.removeSource(1);
      const newCount = await this.getSourceCount();
      if (newCount >= count) {
        console.log('Warning: Source count did not decrease, stopping clear');
        break;
      }
      count = newCount;
    }
  }

  async enableFilter(filterId: string): Promise<void> {
    const filterPanel = this.page.locator('filter-panel');
    const checkbox = filterPanel.locator(`[data-testid="filter-checkbox-${filterId}"]`);
    if (!await checkbox.isChecked()) {
      await checkbox.check();
      await this.page.waitForTimeout(500);
    }
  }

  async disableFilter(filterId: string): Promise<void> {
    const filterPanel = this.page.locator('filter-panel');
    const checkbox = filterPanel.locator(`[data-testid="filter-checkbox-${filterId}"]`);
    if (await checkbox.isChecked()) {
      await checkbox.uncheck();
      await this.page.waitForTimeout(500);
    }
  }

  async selectFilterParameter(filterId: string, paramId: string, value: string): Promise<void> {
    const filterPanel = this.page.locator('filter-panel');
    await filterPanel.locator(`[data-testid="filter-parameter-${filterId}-${paramId}-${value}"]`).click();
    await this.page.waitForTimeout(500);
  }

  async selectResolution(resolution: string): Promise<void> {
    await this.page.selectOption('[data-testid="resolution-select"]', resolution);
    await this.page.waitForTimeout(500);
  }

  async getSourceCount(): Promise<number> {
    return await this.page.locator('[data-testid="video-source-card"]').count();
  }

  async getGridColumns(): Promise<number> {
    const grid = this.page.locator('[data-testid="video-grid"]');
    const columns = await grid.evaluate(el => {
      const style = window.getComputedStyle(el);
      return style.gridTemplateColumns.split(' ').length;
    });
    return columns;
  }

  async isFilterEnabled(filterId: string): Promise<boolean> {
    const filterPanel = this.page.locator('filter-panel');
    return await filterPanel.locator(`[data-testid="filter-checkbox-${filterId}"]`).isChecked();
  }

  async getSelectedParameter(filterId: string, paramId: string): Promise<string | null> {
    const filterPanel = this.page.locator('filter-panel');
    const radios = filterPanel.locator(`input[name="${filterId}-${paramId}"]`);
    const count = await radios.count();
    
    for (let i = 0; i < count; i++) {
      const radio = radios.nth(i);
      if (await radio.isChecked()) {
        return await radio.getAttribute('value');
      }
    }
    return null;
  }

  async openDrawer(): Promise<void> {
    await this.page.click('[data-testid="add-input-fab"]');
    await this.page.waitForSelector('[data-testid="source-drawer"]', { state: 'visible' });
  }

  async closeDrawer(): Promise<void> {
    await this.page.click('[data-testid="drawer-close"]');
    await this.page.waitForTimeout(300);
  }

  async isDrawerOpen(): Promise<boolean> {
    const drawer = this.page.locator('[data-testid="source-drawer"]');
    const classList = await drawer.evaluate(el => el.className);
    return classList.includes('show');
  }

  expectConsoleLogContains(pattern: string | RegExp): void {
    const found = this.consoleLogs.some(log => {
      if (typeof pattern === 'string') {
        return log.text.includes(pattern);
      }
      return pattern.test(log.text);
    });
    expect(found).toBeTruthy();
  }

  async waitForImageLoad(sourceNumber: number, timeout = 5000): Promise<void> {
    await this.page.waitForFunction(
      (num) => {
        const card = document.querySelector(`[data-source-number="${num}"]`);
        const img = card?.querySelector('img');
        return img && img.complete && img.naturalHeight > 0;
      },
      sourceNumber,
      { timeout }
    );
  }
}

