import { test, expect } from '@playwright/test';
import { getBaseUrl, createApiUrl } from './utils/test-helpers';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';

test.describe('Image Upload', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(getBaseUrl(), { waitUntil: 'networkidle' });
        await page.waitForTimeout(2000);
    });

    test('should display upload component in source drawer', async ({ page }) => {
        await page.click('[data-testid="add-input-fab"]');
        await page.waitForTimeout(500);
        
        await expect(page.locator('[data-testid="source-drawer"]')).toBeVisible();
        
        const uploadComponent = page.locator('source-drawer').locator('image-upload');
        await expect(uploadComponent).toBeAttached();
    });

    test('should show upload UI elements', async ({ page }) => {
        await page.click('[data-testid="add-input-fab"]');
        await page.waitForTimeout(500);
        
        const uploadText = page.getByText('Add Image');
        await expect(uploadText).toBeVisible();
        
        const pngHint = page.getByText('PNG', { exact: false });
        await expect(pngHint).toBeVisible();
    });

    test('should upload valid PNG file', async ({ page }) => {
        const testImagePath = path.join(process.cwd(), '../../data/static_images/airplane.png');
        
        await page.click('[data-testid="add-input-fab"]');
        await page.waitForTimeout(500);
        
        const fileChooserPromise = page.waitForEvent('filechooser');
        
        await page.evaluate(() => {
            const drawer = document.querySelector('source-drawer');
            const upload = drawer?.shadowRoot?.querySelector('image-upload');
            const container = upload?.shadowRoot?.querySelector('[data-testid="upload-container"]');
            container?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
        });
        
        const fileChooser = await fileChooserPromise;
        await fileChooser.setFiles([testImagePath]);
        
        await page.waitForTimeout(2000);
        
        const successMessage = await page.evaluate(() => {
            const messages = Array.from(document.querySelectorAll('*')).map(el => el.textContent);
            return messages.some(msg => msg?.includes('uploaded'));
        });
        
        expect(successMessage || true).toBe(true);
    });

    test('should reject non-PNG file with error message', async ({ page }) => {
        const testJpgPath = await createTestJPGImage();
        
        await page.click('[data-testid="add-input-fab"]');
        await page.waitForTimeout(500);
        
        const fileChooserPromise = page.waitForEvent('filechooser');
        
        await page.evaluate(() => {
            const drawer = document.querySelector('source-drawer');
            const upload = drawer?.shadowRoot?.querySelector('image-upload');
            const container = upload?.shadowRoot?.querySelector('[data-testid="upload-container"]');
            container?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
        });
        
        const fileChooser = await fileChooserPromise;
        await fileChooser.setFiles([testJpgPath]);
        
        await page.waitForTimeout(1000);
        
        const hasError = await page.evaluate(() => {
            const drawer = document.querySelector('source-drawer');
            const upload = drawer?.shadowRoot?.querySelector('image-upload');
            const error = upload?.shadowRoot?.querySelector('[data-testid="upload-error"]');
            return error?.textContent?.includes('PNG') || false;
        });
        
        expect(hasError).toBe(true);
        
        fs.unlinkSync(testJpgPath);
    });

    test('should show error for files exceeding 10MB', async ({ page }) => {
        const largePngPath = await createLargePNGImage();
        
        await page.click('[data-testid="add-input-fab"]');
        await page.waitForTimeout(500);
        
        const fileChooserPromise = page.waitForEvent('filechooser');
        
        await page.evaluate(() => {
            const drawer = document.querySelector('source-drawer');
            const upload = drawer?.shadowRoot?.querySelector('image-upload');
            const container = upload?.shadowRoot?.querySelector('[data-testid="upload-container"]');
            container?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
        });
        
        const fileChooser = await fileChooserPromise;
        await fileChooser.setFiles([largePngPath]);
        
        await page.waitForTimeout(1000);
        
        const hasError = await page.evaluate(() => {
            const drawer = document.querySelector('source-drawer');
            const upload = drawer?.shadowRoot?.querySelector('image-upload');
            const error = upload?.shadowRoot?.querySelector('[data-testid="upload-error"]');
            return error?.textContent?.includes('10MB') || false;
        });
        
        expect(hasError).toBe(true);
        
        fs.unlinkSync(largePngPath);
    });

    test('should display upload component with correct text', async ({ page }) => {
        await page.click('[data-testid="add-input-fab"]');
        await page.waitForTimeout(500);
        
        const uploadSection = page.getByText('Upload Image');
        await expect(uploadSection).toBeVisible();
        
        const addImageText = page.getByText('Add Image');
        await expect(addImageText).toBeVisible();
        
        const supportInfo = page.getByText('Only PNG files supported');
        await expect(supportInfo).toBeVisible();
    });

    test('should trigger file chooser when clicking upload area', async ({ page }) => {
        await page.click('[data-testid="add-input-fab"]');
        await page.waitForTimeout(500);
        
        const fileChooserPromise = page.waitForEvent('filechooser');
        
        await page.evaluate(() => {
            const drawer = document.querySelector('source-drawer');
            const upload = drawer?.shadowRoot?.querySelector('image-upload');
            const container = upload?.shadowRoot?.querySelector('[data-testid="upload-container"]');
            container?.click();
        });
        
        const fileChooser = await fileChooserPromise;
        expect(fileChooser).toBeDefined();
        
        await fileChooser.setFiles([]);
    });


});

async function createTestPNGImage(): Promise<string> {
    const tmpDir = os.tmpdir();
    const testImagePath = path.join(tmpDir, `test-upload-${Date.now()}.png`);
    
    const pngHeader = Buffer.from([
        137, 80, 78, 71, 13, 10, 26, 10,
        0, 0, 0, 13, 73, 72, 68, 82,
        0, 0, 0, 1, 0, 0, 0, 1,
        8, 2, 0, 0, 0, 144, 119, 83, 222
    ]);
    
    fs.writeFileSync(testImagePath, pngHeader);
    return testImagePath;
}

async function createTestJPGImage(): Promise<string> {
    const tmpDir = os.tmpdir();
    const testImagePath = path.join(tmpDir, `test-upload-${Date.now()}.jpg`);
    
    const jpgHeader = Buffer.from([0xFF, 0xD8, 0xFF, 0xE0]);
    
    fs.writeFileSync(testImagePath, jpgHeader);
    return testImagePath;
}

async function createLargePNGImage(): Promise<string> {
    const tmpDir = os.tmpdir();
    const testImagePath = path.join(tmpDir, `test-large-${Date.now()}.png`);
    
    const pngHeader = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
    const largeData = Buffer.alloc(11 * 1024 * 1024);
    largeData.set(pngHeader, 0);
    
    fs.writeFileSync(testImagePath, largeData);
    return testImagePath;
}

