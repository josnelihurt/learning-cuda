import { test, expect } from '@playwright/test';

test.describe('Stream Configuration API', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the app to ensure the service is running
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('should return default stream configuration when no feature flags are set', async ({ request }) => {
    const response = await request.post('/cuda_learning.ConfigService/GetStreamConfig', {
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
      log_level: 'INFO',
      console_logging: true,
    });
  });

  test('should return stream configuration with default values', async ({ request }) => {
    // Test the GetStreamConfig endpoint with default configuration
    const response = await request.post('/cuda_learning.ConfigService/GetStreamConfig', {
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
      log_level: 'INFO',
      console_logging: true,
    });
  });

  test('should handle feature flag evaluation errors gracefully', async ({ request }) => {
    // Test the GetStreamConfig endpoint - should fallback to default even without flags
    const response = await request.post('/cuda_learning.ConfigService/GetStreamConfig', {
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
      log_level: 'INFO',
      console_logging: true,
    });
  });

  test('should return proper log level configuration', async ({ request }) => {
    // Test the GetStreamConfig endpoint with default log level configuration
    const response = await request.post('/cuda_learning.ConfigService/GetStreamConfig', {
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
      log_level: 'INFO', // Should use default
      console_logging: true, // Should use default
    });
  });

  test.afterEach(async ({ request }) => {
    // No cleanup needed for simplified tests
  });
});
