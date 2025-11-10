import { defineConfig, devices } from '@playwright/test';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

// Determinar el entorno (default: development)
const environment = process.env.TEST_ENV || 'development';

// Obtener __dirname en ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Cargar variables de entorno del archivo correspondiente
dotenv.config({ 
  path: path.resolve(__dirname, `.env.${environment}`) 
});

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.PLAYWRIGHT_WORKERS ? parseInt(process.env.PLAYWRIGHT_WORKERS) : (process.env.CI ? 4 : 25),
  timeout: 60000,
  
  reporter: [
    ['html', { outputFolder: '../../.ignore/webserver/web/playwright-report', open: 'never' }],
    ['json', { outputFile: '../../.ignore/webserver/web/test-results/e2e-results.json' }],
    ['junit', { outputFile: '../../.ignore/webserver/web/test-results/e2e-junit.xml' }],
    ['list']
  ],
  
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL,
    trace: process.env.PLAYWRIGHT_TRACE as 'on' | 'off' | 'on-first-retry' | 'retain-on-failure' || 'on-first-retry',
    screenshot: process.env.PLAYWRIGHT_SCREENSHOTS as 'on' | 'off' | 'only-on-failure' || 'only-on-failure',
    video: process.env.PLAYWRIGHT_VIDEO === 'true' ? 'retain-on-failure' : 'off',
    ignoreHTTPSErrors: true, // Siempre true para certificados auto-firmados
    actionTimeout: 10000,
    navigationTimeout: 15000,
  },

  projects: process.env.PLAYWRIGHT_SINGLE_BROWSER === 'true'
    ? [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }]
    : [
        { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
        { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
        { name: 'webkit', use: { ...devices['Desktop Safari'] } },
      ],
});

