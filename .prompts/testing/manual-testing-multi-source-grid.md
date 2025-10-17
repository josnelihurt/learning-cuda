# Multi-Source Video Grid - Manual Testing Suite

## Metadata
- **Feature**: Multi-source video processing grid
- **Version**: 1.0.0
- **Date Created**: 2025-10-17
- **Last Updated**: 2025-10-17
- **Related PRs**: Multi-source grid implementation
- **Prerequisites**: Backend server running, browser with Playwright support

## Overview
Execute the following browser-based tests to verify the multi-source video grid functionality, including dynamic resolution control, independent filter configuration per source, and grid layout management.

## Prerequisites
- Navigate to https://localhost:8443
- Ensure backend server is running
- Accept SSL certificate if prompted
- Open browser console to verify logs

## Test Suite 1: Resolution Control
**Objective:** Verify frontend image scaling works correctly

### Test 1.1: Original Resolution
1. Apply grayscale filter to source 1
2. **Expected Console Output:**
   ```
   Applying filter to source 1 : [grayscale] bt601 resolution: original
   Sending image: 512 x 512 → 512 x 512
   ```
3. Confirm image is processed in grayscale

### Test 1.2: Half Resolution
1. Change resolution dropdown to "Half"
2. **Expected Console Output:**
   ```
   Applying filter to source 1 : [grayscale] bt601 resolution: half
   Sending image: 512 x 512 → 256 x 256
   ```
3. Confirm processed image updates (75% data reduction)

### Test 1.3: Quarter Resolution
1. Change resolution dropdown to "Quarter"
2. **Expected Console Output:**
   ```
   Applying filter to source 1 : [grayscale] bt601 resolution: quarter
   Sending image: 512 x 512 → 128 x 128
   ```
3. Confirm processed image updates (93.75% data reduction)

### Test 1.4: Resolution Persistence
1. Change back to "Original Size"
2. **Expected Console Output:**
   ```
   Sending image: 512 x 512 → 512 x 512
   ```
3. Verify resolution setting persists when switching sources

## Test Suite 2: Multi-Source Management
**Objective:** Verify grid can handle multiple sources up to maximum capacity

### Test 2.1: Add Multiple Sources
1. Click "+ Add Input" button
2. Select "Lena" from drawer
3. Repeat 3 times (total 4 sources)
4. **Expected:**
   - Grid shows 2x2 layout
   - Each source has unique number (1, 2, 3, 4)
   - All sources visible without scrolling
   - Console logs: "Source added to grid: lena-N Total: N"

### Test 2.2: Verify Grid Layout
1. Check that all 4 sources fit in viewport
2. Confirm sources scale proportionally
3. Verify grid maintains 4:3 aspect ratio
4. No horizontal/vertical scrolling required

### Test 2.3: Maximum Capacity
1. Continue adding sources until reaching 9 total
2. **Expected:**
   - Grid shows 3x3 layout
   - All 9 sources visible
   - "+ Add Input" shows warning at capacity
   - Console: "Cannot add more than 9 sources"

### Test 2.4: Grid Layout Transitions
Verify grid adapts layout based on source count:
- 1 source: 1x1 (full screen)
- 2 sources: 1x2 (vertical)
- 3-4 sources: 2x2
- 5-6 sources: 2x3
- 7-9 sources: 3x3

## Test Suite 3: Independent Filter Configuration
**Objective:** Verify each source maintains independent filter and resolution state

### Test 3.1: Configure Source 1
1. Select source 1 (click on card #1)
2. Apply grayscale with BT.601 algorithm
3. Set resolution to "Original Size"
4. **Expected Console:**
   ```
   Applying filter to source 1 : [grayscale] bt601 resolution: original
   Sending image: 512 x 512 → 512 x 512
   ```

### Test 3.2: Configure Source 2 (Different Settings)
1. Select source 2
2. Verify panel shows NO filters initially
3. Apply grayscale with BT.709 algorithm
4. Set resolution to "Half"
5. **Expected Console:**
   ```
   Applying filter to source 2 : [grayscale] bt709 resolution: half
   Sending image: 512 x 512 → 256 x 256
   ```

### Test 3.3: Configure Source 3 (Different Algorithm)
1. Select source 3
2. Apply grayscale with "Average" algorithm
3. Set resolution to "Quarter"
4. **Expected Console:**
   ```
   Applying filter to source 3 : [grayscale] average resolution: quarter
   Sending image: 512 x 512 → 128 x 128
   ```

### Test 3.4: Configure Source 4 (Different Algorithm)
1. Select source 4
2. Apply grayscale with "Lightness" algorithm
3. Keep resolution as "Original Size"
4. **Expected Console:**
   ```
   Applying filter to source 4 : [grayscale] lightness resolution: original
   Sending image: 512 x 512 → 512 x 512
   ```

## Test Suite 4: Panel Synchronization
**Objective:** Verify UI panel syncs correctly with selected source state

### Test 4.1: Verify Source 1 Panel State
1. Click on source 1 card
2. **Expected Panel State:**
   - Selected indicator: "1 Lena"
   - Grayscale checkbox: ✓ checked
   - Algorithm radio: BT.601 selected
   - Resolution dropdown: "Original Size"

### Test 4.2: Verify Source 2 Panel State
1. Click on source 2 card
2. **Expected Panel State:**
   - Selected indicator: "2 Lena"
   - Grayscale checkbox: ✓ checked
   - Algorithm radio: BT.709 selected
   - Resolution dropdown: "Half"

### Test 4.3: Verify Source 3 Panel State
1. Click on source 3 card
2. **Expected Panel State:**
   - Selected indicator: "3 Lena"
   - Grayscale checkbox: ✓ checked
   - Algorithm radio: Average selected
   - Resolution dropdown: "Quarter"

### Test 4.4: Verify Source 4 Panel State
1. Click on source 4 card
2. **Expected Panel State:**
   - Selected indicator: "4 Lena"
   - Grayscale checkbox: ✓ checked
   - Algorithm radio: Lightness selected
   - Resolution dropdown: "Original Size"

### Test 4.5: Rapid Source Switching
1. Quickly cycle through all sources (1→2→3→4→1)
2. Verify panel updates instantly each time
3. No lag or incorrect state display
4. Console shows proper source selection logs

## Test Suite 5: Filter Toggle and Changes
**Objective:** Verify filters can be enabled/disabled and changed dynamically

### Test 5.1: Disable Filter
1. Select any source with grayscale active
2. Uncheck grayscale checkbox
3. **Expected:**
   - Console: `[none] bt601 resolution: ...`
   - Image returns to color
   - Processed within 2 seconds

### Test 5.2: Re-enable Filter
1. Check grayscale checkbox again
2. **Expected:**
   - Filter re-applies immediately
   - Image shows grayscale
   - Console confirms filter application

### Test 5.3: Change Algorithm While Active
1. With grayscale enabled, click different algorithm radios
2. Test all algorithms: BT.601 → BT.709 → Average → Lightness → Luminosity
3. **Expected:**
   - Each change triggers immediate processing
   - Console shows new algorithm each time
   - Visual differences observable between algorithms
   - No errors or delays

### Test 5.4: Change Resolution While Filtered
1. Keep grayscale active
2. Change resolution: Original → Half → Quarter → Original
3. **Expected:**
   - Each change reprocesses with new resolution
   - Console shows correct dimensions
   - Filter remains active throughout

## Test Suite 6: Source Removal
**Objective:** Verify sources can be removed without affecting others

### Test 6.1: Remove Middle Source
1. Hover over source 2 card
2. Click the "×" close button in top-right corner
3. **Expected:**
   - Source 2 removed from grid
   - Grid reflows to 3 sources (2 columns × 2 rows, one empty)
   - Remaining sources keep their numbers: 1, 3, 4
   - Console: "Source removed from grid: lena-2 Remaining: 3"

### Test 6.2: Verify State Preservation After Removal
1. Select source 3
2. **Expected:**
   - Panel shows "3 Lena"
   - Grayscale still checked
   - Algorithm still "Average"
   - Resolution still "Quarter"
   - All settings preserved

### Test 6.3: Remove Multiple Sources
1. Remove sources sequentially until only 1 remains
2. **Expected:**
   - Grid layout adjusts each time
   - Remaining source maintains all settings
   - No console errors
   - WebSocket connections close properly

### Test 6.4: Remove and Re-add Source
1. Remove source 3
2. Add new "Lena" source
3. **Expected:**
   - New source gets next available number
   - New source has default settings (no filters, original resolution)
   - Independent from removed source

## Test Suite 7: Drawer Functionality
**Objective:** Verify source selection drawer works correctly

### Test 7.1: Open Drawer
1. Click "+ Add Input" FAB button
2. **Expected:**
   - Drawer slides in from right smoothly
   - Available sources listed: "Lena" (static) and "Camera"
   - Sources show type badges
   - Background dims slightly

### Test 7.2: Select Source from Drawer
1. Click on "Lena" in drawer
2. **Expected:**
   - Drawer auto-closes immediately
   - New source appears in grid
   - Console: "Source selected: lena static"
   - Console: "Source added to grid: lena-N Total: N"

### Test 7.3: Close Drawer Without Selection
1. Open drawer
2. Press ESC key or click outside drawer
3. **Expected:**
   - Drawer closes smoothly
   - No source added
   - Grid remains unchanged

### Test 7.4: Verify Duplicate Sources Allowed
1. Add "Lena" 3 times consecutively
2. **Expected:**
   - Each instance gets unique ID (lena-1, lena-2, lena-3)
   - All instances work independently
   - Each can have different filters/resolution
   - Console confirms unique IDs

## Test Suite 8: WebSocket Management
**Objective:** Verify WebSocket connections are properly managed

### Test 8.1: Verify Individual WebSockets
1. Add 4 sources
2. Open browser Network tab
3. **Expected:**
   - 4 separate WebSocket connections visible
   - Each connection labeled distinctly
   - All connections show "connected" status
   - Console: "WebSocket connected" × 4

### Test 8.2: WebSocket Cleanup on Removal
1. Remove source 2
2. Check Network tab
3. **Expected:**
   - Source 2's WebSocket closes
   - Other 3 WebSockets remain open
   - No connection leaks

### Test 8.3: WebSocket Stability
1. Apply filters to multiple sources simultaneously
2. Change resolutions across sources
3. **Expected:**
   - All WebSockets remain stable
   - No disconnections
   - Messages processed correctly
   - Status bar shows "Connected" (green)

## Expected Console Output Pattern

For each filter application, expect this pattern:
```
Card selected: lena-N N
Applying filter to source N : [filter_type] algorithm resolution: size
Sending image: W x H → target_W x target_H
Filter applied, updating image for source N
Image loaded for source: name target_W x target_H
```

## Success Criteria

All tests must pass with following conditions:
- ✓ All resolution modes scale images correctly (original/half/quarter)
- ✓ Each source maintains completely independent state
- ✓ Panel syncs instantly when switching between sources
- ✓ Filters apply and remove without errors or delays
- ✓ Grid layout adapts dynamically for 1-9 sources
- ✓ No console errors during any operations
- ✓ WebSocket connections remain stable throughout
- ✓ Source removal doesn't affect other sources
- ✓ Drawer opens/closes smoothly without glitches
- ✓ No memory leaks after multiple add/remove cycles
- ✓ Performance remains responsive with 9 sources
- ✓ Visual feedback is immediate for all actions

## Known Issues / Limitations
- Maximum 9 sources (by design)
- Browser may limit concurrent WebSocket connections
- TLS certificate warnings in development mode (expected)

## Performance Benchmarks
- Filter application: < 2 seconds per source
- Source addition: < 500ms
- Source removal: < 200ms
- Panel sync: < 100ms
- Grid reflow: < 300ms

## Troubleshooting

### Issue: Filters not applying
- Check WebSocket connection status
- Verify backend server is running
- Check console for error messages

### Issue: Panel not syncing
- Refresh page and retry
- Check browser console for JavaScript errors

### Issue: Grid layout broken
- Verify window size is adequate (min 1024×768)
- Check CSS loaded correctly
- Try zooming to 100%

## Related Files
- Feature implementation: `webserver/web/src/components/video-grid.ts`
- Filter panel: `webserver/web/src/components/filter-panel.ts`
- Source drawer: `webserver/web/src/components/source-drawer.ts`
- WebSocket service: `webserver/web/src/services/websocket-service.ts`
- BDD tests: `integration/tests/acceptance/features/input_sources.feature`

