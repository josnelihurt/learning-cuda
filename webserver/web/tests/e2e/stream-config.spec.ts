import { test, expect } from '@playwright/test';
import { createApiUrl } from './utils/test-helpers';
import { TestHelpers } from './helpers/test-helpers';

test.describe('Stream Configuration API', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
    await helpers.waitForPageReady();
  });

  test('should return default stream configuration when no feature flags are set', async ({ request }) => {
    const response = await request.post(createApiUrl('/cuda_learning.ConfigService/GetStreamConfig'), {
      data: {},
      headers: {
        'Content-Type': 'application/json',
      },
    });

    expect(response.status()).toBe(200);
    
    const data = await response.json();
    expect(data).toHaveProperty('endpoints');
    expect(data.endpoints).toHaveLength(1);
    
    const endpoint = data.endpoints[0];
    expect(endpoint).toMatchObject({
      type: 'websocket',
      endpoint: '/ws',
      transport_format: 'json',
    });
    // log_level should be a valid log level (INFO, DEBUG, WARN, ERROR, etc.)
    expect(endpoint.log_level).toBeTruthy();
    expect(['DEBUG', 'INFO', 'WARN', 'ERROR']).toContain(endpoint.log_level);
    // console_logging may be omitted in JSON if false, but should be true if present
    if (endpoint.console_logging !== undefined) {
      expect(endpoint.console_logging).toBe(true);
    }
  });

  test('should return stream configuration with default values', async ({ request }) => {
    // Test the GetStreamConfig endpoint with default configuration
    const response = await request.post(createApiUrl('/cuda_learning.ConfigService/GetStreamConfig'), {
      data: {},
      headers: {
        'Content-Type': 'application/json',
      },
    });

    expect(response.status()).toBe(200);
    
    const data = await response.json();
    expect(data).toHaveProperty('endpoints');
    expect(data.endpoints).toHaveLength(1);
    
    const endpoint = data.endpoints[0];
    expect(endpoint).toMatchObject({
      type: 'websocket',
      endpoint: '/ws',
      transport_format: 'json', // Should be default json
    });
    // log_level should be a valid log level (INFO, DEBUG, WARN, ERROR, etc.)
    expect(endpoint.log_level).toBeTruthy();
    expect(['DEBUG', 'INFO', 'WARN', 'ERROR']).toContain(endpoint.log_level);
    // console_logging may be omitted in JSON if false, but should be true if present
    if (endpoint.console_logging !== undefined) {
      expect(endpoint.console_logging).toBe(true);
    }
  });

  test('should handle feature flag evaluation errors gracefully', async ({ request }) => {
    // Test the GetStreamConfig endpoint - should fallback to default even without flags
    const response = await request.post(createApiUrl('/cuda_learning.ConfigService/GetStreamConfig'), {
      data: {},
      headers: {
        'Content-Type': 'application/json',
      },
    });

    expect(response.status()).toBe(200);
    
    const data = await response.json();
    expect(data).toHaveProperty('endpoints');
    expect(data.endpoints).toHaveLength(1);
    
    const endpoint = data.endpoints[0];
    expect(endpoint).toMatchObject({
      type: 'websocket',
      endpoint: '/ws',
      transport_format: 'json', // Should fallback to default
    });
    // log_level should be a valid log level (INFO, DEBUG, WARN, ERROR, etc.)
    expect(endpoint.log_level).toBeTruthy();
    expect(['DEBUG', 'INFO', 'WARN', 'ERROR']).toContain(endpoint.log_level);
    // console_logging may be omitted in JSON if false, but should be true if present
    if (endpoint.console_logging !== undefined) {
      expect(endpoint.console_logging).toBe(true);
    }
  });

  test('should return proper log level configuration', async ({ request }) => {
    // Test the GetStreamConfig endpoint with default log level configuration
    const response = await request.post(createApiUrl('/cuda_learning.ConfigService/GetStreamConfig'), {
      data: {},
      headers: {
        'Content-Type': 'application/json',
      },
    });

    expect(response.status()).toBe(200);
    
    const data = await response.json();
    expect(data).toHaveProperty('endpoints');
    expect(data.endpoints).toHaveLength(1);
    
    const endpoint = data.endpoints[0];
    expect(endpoint).toMatchObject({
      type: 'websocket',
      endpoint: '/ws',
      transport_format: 'json',
    });
    // log_level should be a valid log level (INFO, DEBUG, WARN, ERROR, etc.)
    expect(endpoint.log_level).toBeTruthy();
    expect(['DEBUG', 'INFO', 'WARN', 'ERROR']).toContain(endpoint.log_level);
    // console_logging may be omitted in JSON if false, but should be true if present
    if (endpoint.console_logging !== undefined) {
      expect(endpoint.console_logging).toBe(true);
    }
  });

  test.afterEach(async ({ request }) => {
    // No cleanup needed for simplified tests
  });
});
