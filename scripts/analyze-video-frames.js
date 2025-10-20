#!/usr/bin/env node
/**
 * Script to analyze frames from sample video and calculate
 * the real change percentage between consecutive frames.
 * 
 * This helps us determine a realistic threshold for E2E tests.
 */

const fs = require('fs');
const { createCanvas, loadImage } = require('canvas');
const path = require('path');

async function analyzeVideoPreview() {
    const previewPath = path.join(__dirname, '../data/video_previews/sample.png');
    
    if (!fs.existsSync(previewPath)) {
        console.error('Error: Video preview not found at:', previewPath);
        process.exit(1);
    }

    console.log('Analyzing video preview:', previewPath);
    console.log('');

    const img = await loadImage(previewPath);
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');

    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, img.width, img.height);
    const pixels = imageData.data;

    console.log(`Dimensions: ${img.width}x${img.height}`);
    console.log(`Total pixels: ${(pixels.length / 4).toLocaleString()}`);
    console.log(`Total bytes: ${pixels.length.toLocaleString()}`);
    console.log('');

    // Simulate backend variation (modify 64x64 pixel region)
    const modifiedPixels = new Uint8ClampedArray(pixels);
    const regionSize = 64;
    const numPixelsToModify = regionSize * regionSize; // 4,096 pixels
    
    // Modify 64x64 region in top-left corner
    for (let y = 0; y < regionSize; y++) {
        for (let x = 0; x < regionSize; x++) {
            const pixelIndex = (y * img.width + x) * 4;
            // Simulate variation with frame number = 50 (half of 0-255 range)
            const variation = 50;
            modifiedPixels[pixelIndex] = Math.max(0, Math.min(255, pixels[pixelIndex] + variation));
            modifiedPixels[pixelIndex + 1] = Math.max(0, Math.min(255, pixels[pixelIndex + 1] + variation));
            modifiedPixels[pixelIndex + 2] = Math.max(0, Math.min(255, pixels[pixelIndex + 2] + variation));
        }
    }

    // Calculate difference
    let differentPixels = 0;
    const threshold = 5;
    const totalPixels = img.width * img.height;

    for (let i = 0; i < pixels.length; i += 4) {
        const diff = Math.abs(modifiedPixels[i] - pixels[i]) +
                    Math.abs(modifiedPixels[i + 1] - pixels[i + 1]) +
                    Math.abs(modifiedPixels[i + 2] - pixels[i + 2]);
        
        if (diff > threshold) {
            differentPixels++;
        }
    }

    const percentDifferent = (differentPixels / totalPixels) * 100;

    console.log('=== SIMULATED VARIATION ANALYSIS ===');
    console.log(`Pixels modified in backend: ${numPixelsToModify}`);
    console.log(`Pixels detected as different: ${differentPixels}`);
    console.log(`Change percentage: ${percentDifferent.toFixed(4)}%`);
    console.log('');

    // Recommendations
    console.log('=== RECOMMENDATIONS ===');
    
    if (percentDifferent < 0.1) {
        console.log('WARNING: Change percentage is very low (<0.1%)');
        console.log('');
        console.log('Options to improve:');
        console.log(`1. Increase modified pixels in backend from ${numPixelsToModify} to ${Math.ceil(totalPixels * 0.01)} (~1% of image)`);
        console.log('2. Increase color variation (currently ±50)');
        console.log('3. Modify a larger region (entire top-left corner)');
    }

    const recommendedThreshold = Math.max(0.001, percentDifferent * 0.5);
    console.log('');
    console.log(`Recommended threshold for E2E test: ${recommendedThreshold.toFixed(4)}%`);
    console.log(`(50% of detected real change to allow error margin)`);
    console.log('');

    // Calculate how many pixels we need to modify for 1% change
    const pixelsFor1Percent = Math.ceil(totalPixels * 0.01);
    console.log(`To achieve 1% visible change, modify ${pixelsFor1Percent.toLocaleString()} pixels`);
    console.log(`Example: modify region of ${Math.ceil(Math.sqrt(pixelsFor1Percent))}x${Math.ceil(Math.sqrt(pixelsFor1Percent))} pixels`);
}

analyzeVideoPreview().catch(err => {
    console.error('Error:', err);
    process.exit(1);
});
