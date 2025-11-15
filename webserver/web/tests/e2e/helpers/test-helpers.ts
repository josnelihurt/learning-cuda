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
    await this.page.waitForTimeout(150);
    
    await this.page.waitForFunction(
      (num) => {
        const grid = document.querySelector('video-grid') as any;
        if (!grid) return false;
        const selected = grid.getSelectedSource();
        return selected && selected.number === num;
      },
      sourceNumber,
      { timeout: 2000 }
    );
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
    await filterPanel.waitFor({ state: 'visible', timeout: 2000 });
    const checkbox = filterPanel.locator(`[data-testid="filter-checkbox-${filterId}"]`);
    await checkbox.waitFor({ state: 'visible', timeout: 2000 });
    return await checkbox.isChecked();
  }

  async getSelectedParameter(filterId: string, paramId: string): Promise<string | null> {
    const filterPanel = this.page.locator('filter-panel');
    await filterPanel.waitFor({ state: 'visible', timeout: 2000 });
    const radios = filterPanel.locator(`input[name="${filterId}-${paramId}"]`);
    await radios.first().waitFor({ state: 'attached', timeout: 2000 });
    const count = await radios.count();
    
    for (let i = 0; i < count; i++) {
      const radio = radios.nth(i);
      if (await radio.isChecked()) {
        return await radio.getAttribute('value');
      }
    }
    return null;
  }

  async setBlurParameter(filterId: string, paramId: string, value: string | number): Promise<void> {
    const filterPanel = this.page.locator('filter-panel');
    await filterPanel.waitFor({ state: 'visible', timeout: 2000 });
    const paramControl = filterPanel.locator(`[data-testid="filter-parameter-${filterId}-${paramId}"]`);
    await paramControl.waitFor({ state: 'visible', timeout: 2000 });
    
    const tagName = await paramControl.evaluate(el => el.tagName.toLowerCase());
    
    if (tagName === 'input') {
      const inputType = await paramControl.getAttribute('type');
      if (inputType === 'range' || inputType === 'number') {
        await paramControl.fill(value.toString());
        await paramControl.dispatchEvent('input');
      } else if (inputType === 'checkbox') {
        const isChecked = value === true || value === 'true';
        if (isChecked) {
          await paramControl.check();
        } else {
          await paramControl.uncheck();
        }
      }
    } else if (tagName === 'select') {
      await paramControl.selectOption(value.toString());
    }
    
    await this.page.waitForTimeout(300);
  }

  async getBlurParameter(filterId: string, paramId: string): Promise<string | null> {
    const filterPanel = this.page.locator('filter-panel');
    await filterPanel.waitFor({ state: 'visible', timeout: 2000 });
    const paramControl = filterPanel.locator(`[data-testid="filter-parameter-${filterId}-${paramId}"]`);
    await paramControl.waitFor({ state: 'visible', timeout: 2000 });
    
    const tagName = await paramControl.evaluate(el => el.tagName.toLowerCase());
    
    if (tagName === 'input') {
      const inputType = await paramControl.getAttribute('type');
      if (inputType === 'range' || inputType === 'number') {
        return await paramControl.inputValue();
      } else if (inputType === 'checkbox') {
        const checked = await paramControl.isChecked();
        return checked ? 'true' : 'false';
      }
    } else if (tagName === 'select') {
      return await paramControl.inputValue();
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

  async getSourceFilters(sourceNumber: number): Promise<Array<{ id: string; parameters: Record<string, string> }> | null> {
    return await this.page.evaluate((num) => {
      const grid = document.querySelector('video-grid') as any;
      if (!grid || typeof grid.getSources !== 'function') {
        return null;
      }
      const source = grid.getSources().find((s: any) => s.number === num);
      if (!source) {
        return null;
      }
      if (!Array.isArray(source.filters)) {
        return [];
      }
      return source.filters.map((filter: any) => ({
        id: filter?.id,
        parameters: { ...(filter?.parameters || {}) },
      }));
    }, sourceNumber);
  }

  async waitForSourceFilterParameter(
    sourceNumber: number,
    filterId: string,
    paramId: string,
    expectedValue: string,
    timeout = 5000
  ): Promise<void> {
    await this.page.waitForFunction(
      ({ sourceNumber, filterId, paramId, expectedValue }) => {
        const grid = document.querySelector('video-grid') as any;
        if (!grid || typeof grid.getSources !== 'function') {
          return false;
        }
        const source = grid.getSources().find((s: any) => s.number === sourceNumber);
        if (!source || !Array.isArray(source.filters)) {
          return false;
        }
        const filter = source.filters.find((f: any) => f && f.id === filterId);
        if (!filter || !filter.parameters) {
          return false;
        }
        return filter.parameters[paramId] === expectedValue;
      },
      { sourceNumber, filterId, paramId, expectedValue },
      { timeout }
    );
  }

  async expectSourceFilterParameter(
    sourceNumber: number,
    filterId: string,
    paramId: string,
    expectedValue: string,
    timeout = 5000
  ): Promise<void> {
    await this.waitForSourceFilterParameter(sourceNumber, filterId, paramId, expectedValue, timeout);
    const filters = await this.getSourceFilters(sourceNumber);
    const filter = filters?.find((f) => f.id === filterId);
    expect(filter?.parameters?.[paramId]).toBe(expectedValue);
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

