#!/usr/bin/env node
/**
 * TODO: Move this script to integration tests and convert video to golden test data
 * for automated E2E validation. Store extracted frames in test fixtures directory.
 * Script to analyze real frames from sample.mp4 video
 * Extracts frames at seconds 10, 15 and 30 and calculates pixel differences
 * to determine realistic thresholds for E2E tests.
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const { createCanvas, loadImage } = require('canvas');
const crypto = require('crypto');

const VIDEO_PATH = path.join(__dirname, '../data/videos/sample.mp4');
const OUTPUT_DIR = path.join(__dirname, '../.tmp/video-frames');

// Create temporary directory
if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

async function extractFrame(videoPath, timestamp, outputPath) {
    console.log(`Extracting frame at ${timestamp}s...`);
    try {
        // Use ffmpeg to extract exact frame
        execSync(
            `ffmpeg -y -ss ${timestamp} -i "${videoPath}" -vframes 1 -q:v 2 "${outputPath}" 2>/dev/null`,
            { stdio: 'pipe' }
        );
        return true;
    } catch (error) {
        console.error(`Error extracting frame at ${timestamp}s:`, error.message);
        return false;
    }
}

async function getImageBytes(imagePath) {
    const img = await loadImage(imagePath);
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, img.width, img.height);
    return {
        data: imageData.data,
        width: img.width,
        height: img.height
    };
}

function calculatePixelDifference(bytes1, bytes2, threshold = 5) {
    if (bytes1.length !== bytes2.length) {
        throw new Error('Image sizes do not match');
    }

    let differentPixels = 0;
    const totalPixels = bytes1.length / 4;

    for (let i = 0; i < bytes1.length; i += 4) {
        const r1 = bytes1[i], g1 = bytes1[i + 1], b1 = bytes1[i + 2];
        const r2 = bytes2[i], g2 = bytes2[i + 1], b2 = bytes2[i + 2];

        if (Math.abs(r1 - r2) > threshold || Math.abs(g1 - g2) > threshold || Math.abs(b1 - b2) > threshold) {
            differentPixels++;
        }
    }

    return {
        differentPixels,
        totalPixels,
        percentDifferent: (differentPixels / totalPixels) * 100
    };
}

function calculateHash(bytes) {
    return crypto.createHash('sha256').update(Buffer.from(bytes)).digest('hex');
}

async function analyzeVideo() {
    console.log('=== REAL VIDEO ANALYSIS ===\n');
    console.log(`Video: ${VIDEO_PATH}`);
    console.log('');

    if (!fs.existsSync(VIDEO_PATH)) {
        console.error('Error: Video not found');
        process.exit(1);
    }

    // Extract frames at specific timestamps
    const timestamps = [10, 15, 30];
    const frames = {};

    for (const ts of timestamps) {
        const framePath = path.join(OUTPUT_DIR, `frame_${ts}s.png`);
        const success = await extractFrame(VIDEO_PATH, ts, framePath);
        
        if (!success) {
            console.error(`Could not extract frame at ${ts}s`);
            continue;
        }

        const imageData = await getImageBytes(framePath);
        frames[ts] = imageData;
        
        console.log(`Frame ${ts}s: ${imageData.width}x${imageData.height} (${imageData.data.length.toLocaleString()} bytes)`);
    }

    console.log('\n=== FRAME COMPARISONS ===\n');

    // Compare 10s vs 15s (5 second interval)
    if (frames[10] && frames[15]) {
        const diff1 = calculatePixelDifference(frames[10].data, frames[15].data);
        const hash10 = calculateHash(frames[10].data);
        const hash15 = calculateHash(frames[15].data);

        console.log('Frame 10s vs Frame 15s (5 second difference):');
        console.log(`   Hash 10s: ${hash10.substring(0, 16)}...`);
        console.log(`   Hash 15s: ${hash15.substring(0, 16)}...`);
        console.log(`   Total pixels: ${diff1.totalPixels.toLocaleString()}`);
        console.log(`   Different pixels: ${diff1.differentPixels.toLocaleString()}`);
        console.log(`   Change percentage: ${diff1.percentDifferent.toFixed(4)}%`);
        console.log('');
    }

    // Compare 10s vs 30s (20 second interval)
    if (frames[10] && frames[30]) {
        const diff2 = calculatePixelDifference(frames[10].data, frames[30].data);
        const hash10 = calculateHash(frames[10].data);
        const hash30 = calculateHash(frames[30].data);

        console.log('Frame 10s vs Frame 30s (20 second difference):');
        console.log(`   Hash 10s: ${hash10.substring(0, 16)}...`);
        console.log(`   Hash 30s: ${hash30.substring(0, 16)}...`);
        console.log(`   Total pixels: ${diff2.totalPixels.toLocaleString()}`);
        console.log(`   Different pixels: ${diff2.differentPixels.toLocaleString()}`);
        console.log(`   Change percentage: ${diff2.percentDifferent.toFixed(4)}%`);
        console.log('');
    }

    // Compare 15s vs 30s (15 second interval)
    if (frames[15] && frames[30]) {
        const diff3 = calculatePixelDifference(frames[15].data, frames[30].data);
        const hash15 = calculateHash(frames[15].data);
        const hash30 = calculateHash(frames[30].data);

        console.log('Frame 15s vs Frame 30s (15 second difference):');
        console.log(`   Hash 15s: ${hash15.substring(0, 16)}...`);
        console.log(`   Hash 30s: ${hash30.substring(0, 16)}...`);
        console.log(`   Total pixels: ${diff3.totalPixels.toLocaleString()}`);
        console.log(`   Different pixels: ${diff3.differentPixels.toLocaleString()}`);
        console.log(`   Change percentage: ${diff3.percentDifferent.toFixed(4)}%`);
        console.log('');
    }

    // Short interval analysis (1 second - similar to E2E test)
    console.log('=== SHORT INTERVAL ANALYSIS (1 SECOND) ===\n');
    
    const frame10_0 = path.join(OUTPUT_DIR, 'frame_10.0s.png');
    const frame10_1 = path.join(OUTPUT_DIR, 'frame_10.1s.png');
    
    await extractFrame(VIDEO_PATH, 10.0, frame10_0);
    await extractFrame(VIDEO_PATH, 11.0, frame10_1);
    
    const img10_0 = await getImageBytes(frame10_0);
    const img10_1 = await getImageBytes(frame10_1);
    
    const shortDiff = calculatePixelDifference(img10_0.data, img10_1.data);
    const hash10_0 = calculateHash(img10_0.data);
    const hash10_1 = calculateHash(img10_1.data);
    
    console.log('Frame 10.0s vs Frame 11.0s (1 second difference):');
    console.log(`   Hash 10.0s: ${hash10_0.substring(0, 16)}...`);
    console.log(`   Hash 11.0s: ${hash10_1.substring(0, 16)}...`);
    console.log(`   Different pixels: ${shortDiff.differentPixels.toLocaleString()}`);
    console.log(`   Change percentage: ${shortDiff.percentDifferent.toFixed(4)}%`);
    console.log('');

    // Recommendations
    console.log('=== E2E TEST RECOMMENDATIONS ===\n');
    
    const minPercent = Math.min(
        diff1?.percentDifferent || 100,
        diff2?.percentDifferent || 100,
        diff3?.percentDifferent || 100,
        shortDiff.percentDifferent
    );
    
    const recommendedThreshold = Math.max(0.1, minPercent * 0.3); // 30% of minimum
    
    console.log(`Minimum change detected: ${minPercent.toFixed(4)}%`);
    console.log(`Recommended threshold: ${recommendedThreshold.toFixed(4)}%`);
    console.log(`  (30% of minimum change for robust margin)`);
    console.log('');
    
    if (shortDiff.percentDifferent < 1.0) {
        console.log('WARNING: Video has less than 1% change in 1 second');
        console.log('   This is normal for videos with slow movement');
        console.log('   Backend should add artificial variation for validation');
    } else {
        console.log('Video has sufficient natural variation for validation');
    }
    
    console.log('\n=== SUGGESTED CONFIGURATION ===\n');
    console.log('For E2E test:');
    console.log(`  - Interval: 1 second`);
    console.log(`  - Minimum threshold: ${recommendedThreshold.toFixed(4)}%`);
    console.log(`  - Expected frames: >20 frames`);
    console.log('');
    
    // Cleanup
    console.log(`Extracted frames saved in: ${OUTPUT_DIR}`);
}

analyzeVideo().catch(err => {
    console.error('Error:', err);
    process.exit(1);
});

