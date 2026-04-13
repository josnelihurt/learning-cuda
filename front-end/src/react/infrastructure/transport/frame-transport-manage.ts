import { ReactFrameTransportService } from './ReactFrameTransportService';

/**
 * Singleton instance of ReactFrameTransportService for centralized frame transport access.
 *
 * This follows the same pattern as manageWebRTC (from 04-01-SUMMARY.md) for consistent
 * service access across the React application.
 *
 * Usage:
 * ```typescript
 * import { manageFrameTransport } from '@/react/infrastructure/transport/frame-transport-manage';
 *
 * // Initialize
 * await manageFrameTransport.initialize();
 *
 * // Send frame
 * manageFrameTransport.sendFrame(frameData, filters);
 *
 * // Register callbacks
 * manageFrameTransport.setFrameCallback((frameData) => { /* handle frame */ });
 * manageFrameTransport.setErrorCallback((error) => { /* handle error */ });
 *
 * // Check status
 * const status = manageFrameTransport.getConnectionStatus();
 *
 * // Cleanup
 * manageFrameTransport.close();
 * ```
 */
export const manageFrameTransport = new ReactFrameTransportService();
