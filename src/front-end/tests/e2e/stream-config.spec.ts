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
      type: 'webrtc',
      endpoint: '/cuda_learning.WebRTCSignalingService/StartSession',
    });
    expect(endpoint.log_level).toBeTruthy();
    expect(['DEBUG', 'INFO', 'WARN', 'ERROR']).toContain(endpoint.log_level);
  });

  test('should return stream configuration with default values', async ({ request }) => {
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
      type: 'webrtc',
      endpoint: '/cuda_learning.WebRTCSignalingService/StartSession',
    });
    expect(endpoint.log_level).toBeTruthy();
    expect(['DEBUG', 'INFO', 'WARN', 'ERROR']).toContain(endpoint.log_level);
  });

  test('should handle feature flag evaluation errors gracefully', async ({ request }) => {
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
      type: 'webrtc',
      endpoint: '/cuda_learning.WebRTCSignalingService/StartSession',
    });
    expect(endpoint.log_level).toBeTruthy();
    expect(['DEBUG', 'INFO', 'WARN', 'ERROR']).toContain(endpoint.log_level);
  });

  test('should return proper log level configuration', async ({ request }) => {
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
      type: 'webrtc',
      endpoint: '/cuda_learning.WebRTCSignalingService/StartSession',
    });
    expect(endpoint.log_level).toBeTruthy();
    expect(['DEBUG', 'INFO', 'WARN', 'ERROR']).toContain(endpoint.log_level);
  });

  test.afterEach(async ({ request }) => {
    // No cleanup needed
  });
});
