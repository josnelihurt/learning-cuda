/**
 * Test utilities for E2E tests
 * Provides common functions to avoid code duplication
 */

/**
 * Gets the base URL for API requests based on the current environment
 * @returns The base URL for the current environment
 */
export function getBaseUrl(): string {
  return process.env.PLAYWRIGHT_BASE_URL || 'https://localhost:8443';
}

/**
 * Gets the Flipt API URL based on the current environment
 * @returns The Flipt API URL for the current environment
 */
export function getFliptUrl(): string {
  const env = process.env.TEST_ENV || 'development';
  
  if (env === 'production') {
    return 'https://flipt-cuda-demo.josnelihurt.me';
  } else if (env === 'staging') {
    return 'https://flipt.localhost';
  } else {
    return 'http://localhost:8081';
  }
}

/**
 * Checks if the current environment is production
 * @returns True if running in production environment
 */
export function isProduction(): boolean {
  return process.env.TEST_ENV === 'production';
}

/**
 * Gets the minimum number of frames expected for video tests
 * More lenient in production due to potential performance differences
 * @returns Minimum number of frames expected
 */
export function getMinVideoFrames(): number {
  return isProduction() ? 5 : 20;
}

/**
 * Gets the expected Flipt URL pattern for dashboard tests
 * @returns The expected URL pattern for Flipt dashboard
 */
export function getFliptDashboardUrlPattern(): string {
  const env = process.env.TEST_ENV || 'development';
  
  if (env === 'production') {
    return '/flipt';
  } else if (env === 'staging') {
    return 'flipt.localhost';
  } else {
    return 'localhost:8081';
  }
}

/**
 * Creates a full API endpoint URL
 * @param endpoint The API endpoint path
 * @returns The full URL for the API endpoint
 */
export function createApiUrl(endpoint: string): string {
  const baseUrl = getBaseUrl();
  return `${baseUrl}${endpoint}`;
}

/**
 * Creates a full Flipt API endpoint URL
 * @param endpoint The Flipt API endpoint path
 * @returns The full URL for the Flipt API endpoint
 */
export function createFliptApiUrl(endpoint: string): string {
  const fliptUrl = getFliptUrl();
  return `${fliptUrl}${endpoint}`;
}

/**
 * Gets the appropriate curl options for the current environment
 * @returns Curl options string for the current environment
 */
export function getCurlOptions(): string {
  const env = process.env.TEST_ENV || 'development';
  return (env === 'production' || env === 'staging') ? '-k' : '';
}
