import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: !process.env.CI,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 2 : undefined,
  timeout: 30000,
  
  reporter: [
    ['html', { outputFolder: '.ignore/playwright-report', open: 'never' }],
    ['json', { outputFile: '.ignore/test-results/e2e-results.json' }],
    ['junit', { outputFile: '.ignore/test-results/e2e-junit.xml' }],
    ['list']
  ],
  
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL || 'https://localhost:8443',
    trace: process.env.PLAYWRIGHT_TRACE === 'true' ? 'on' : 'off',
    screenshot: process.env.PLAYWRIGHT_SCREENSHOTS === 'true' ? 'only-on-failure' : 'off',
    video: process.env.PLAYWRIGHT_VIDEO === 'true' ? 'retain-on-failure' : 'off',
    ignoreHTTPSErrors: true,
  },

  projects: process.env.PLAYWRIGHT_SINGLE_BROWSER === 'true'
    ? [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }]
    : [
        { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
        { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
        { name: 'webkit', use: { ...devices['Desktop Safari'] } },
      ],
});

