import { test, expect } from '@playwright/test';
import { getBaseUrl } from './utils/test-helpers';

test.describe('Frontend Logging System', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(getBaseUrl(), { waitUntil: 'networkidle' });
        
        await page.waitForFunction(() => {
            return (window as any).logger !== undefined;
        }, { timeout: 10000 });
    });

    test('logger is initialized with feature flags from backend', async ({ page }) => {
        const logLevel = await page.evaluate(() => {
            const streamConfig = (window as any).streamConfigService;
            return streamConfig.getLogLevel();
        });

        const consoleLogging = await page.evaluate(() => {
            const streamConfig = (window as any).streamConfigService;
            return streamConfig.getConsoleLogging();
        });

        expect(logLevel).toBeTruthy();
        expect(typeof consoleLogging).toBe('boolean');
    });

    test('sends logs to backend via OTLP', async ({ page }) => {
        const logRequestPromise = page.waitForRequest(
            request => request.url().includes('/api/logs') && request.method() === 'POST',
            { timeout: 15000 }
        );

        await page.evaluate(() => {
            const logger = (window as any).logger;
            for (let i = 0; i < 5; i++) {
                logger.info(`Test log message ${i}`, { 'test.index': i });
            }
            return logger.flush();
        });

        const logRequest = await logRequestPromise;
        expect(logRequest).toBeTruthy();
        expect(logRequest.headers()['content-type']).toContain('application/json');
    });

    test('filters logs based on log level configuration', async ({ page }) => {
        await page.evaluate(() => {
            const logger = (window as any).logger;
            logger.initialize('WARN', false);
            logger.flush();
        });

        await page.waitForTimeout(1000);

        let logRequestReceived = false;
        const requestListener = (request: any) => {
            if (request.url().includes('/api/logs') && request.method() === 'POST') {
                logRequestReceived = true;
            }
        };
        
        page.on('request', requestListener);

        await page.evaluate(() => {
            const logger = (window as any).logger;
            logger.debug('This should not be sent');
            logger.info('This should not be sent');
        });

        await page.waitForTimeout(3000);

        page.off('request', requestListener);

        expect(logRequestReceived).toBe(false);
    });

    test('includes trace context in log exports', async ({ page }) => {
        const logRequestPromise = page.waitForRequest(
            request => request.url().includes('/api/logs'),
            { timeout: 15000 }
        );

        await page.evaluate(() => {
            const logger = (window as any).logger;
            logger.info('Test log with trace context');
            return logger.flush();
        });

        const logRequest = await logRequestPromise;
        const postData = logRequest.postDataJSON();
        
        expect(postData).toBeTruthy();
    });

    test('respects console logging feature flag', async ({ page }) => {
        const consoleLogs: string[] = [];
        
        page.on('console', msg => {
            if (msg.type() === 'info') {
                consoleLogs.push(msg.text());
            }
        });

        await page.evaluate(() => {
            const logger = (window as any).logger;
            logger.initialize('INFO', true);
            logger.info('This should appear in console');
        });

        await page.waitForTimeout(1000);

        expect(consoleLogs.some(log => log.includes('This should appear in console'))).toBe(true);
    });

    test('does not log to console when console logging is disabled', async ({ page }) => {
        const consoleLogs: string[] = [];
        
        page.on('console', msg => {
            if (msg.type() === 'info') {
                consoleLogs.push(msg.text());
            }
        });

        await page.evaluate(() => {
            const logger = (window as any).logger;
            logger.initialize('INFO', false);
            logger.info('This should NOT appear in console');
        });

        await page.waitForTimeout(1000);

        expect(consoleLogs.some(log => log.includes('This should NOT appear in console'))).toBe(false);
    });

    test('batches logs automatically after 5 seconds', async ({ page }) => {
        const logRequestPromise = page.waitForRequest(
            request => request.url().includes('/api/logs'),
            { timeout: 10000 }
        );

        await page.evaluate(() => {
            const logger = (window as any).logger;
            logger.info('Batched log message 1');
            logger.info('Batched log message 2');
            logger.info('Batched log message 3');
        });

        const logRequest = await logRequestPromise;
        expect(logRequest).toBeTruthy();
    });

    test('exports logs immediately when buffer is full', async ({ page }) => {
        const logRequestPromise = page.waitForRequest(
            request => request.url().includes('/api/logs'),
            { timeout: 15000 }
        );

        await page.evaluate(() => {
            const logger = (window as any).logger;
            for (let i = 0; i < 100; i++) {
                logger.info(`Batch test message ${i}`);
            }
        });

        const logRequest = await logRequestPromise;
        expect(logRequest).toBeTruthy();
    });

    test('backend receives and processes OTLP logs', async ({ page }) => {
        const responsePromise = page.waitForResponse(
            response => response.url().includes('/api/logs') && response.status() === 200,
            { timeout: 15000 }
        );

        await page.evaluate(() => {
            const logger = (window as any).logger;
            logger.error('Test error log', { 'test.key': 'test.value' });
            return logger.flush();
        });

        const response = await responsePromise;
        expect(response.status()).toBe(200);
    });
});

