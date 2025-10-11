/**
 * StatsManager - Handles real-time statistics tracking
 */
class StatsManager {
    constructor() {
        this.frameCount = 0;
        this.fpsHistory = [];
        this.processingTimes = [];
        this.lastFrameTime = 0;
        
        // DOM elements
        this.statFPS = document.getElementById('statFPS');
        this.statTime = document.getElementById('statTime');
        this.statFrames = document.getElementById('statFrames');
        this.statCamera = document.getElementById('statCamera');
        this.wsIndicator = document.getElementById('wsIndicator');
        this.wsStatus = document.getElementById('wsStatus');
    }
    
    incrementFrameCount() {
        this.frameCount++;
        this.statFrames.textContent = this.frameCount;
    }
    
    updateProcessingStats(processingTime) {
        // Update frame count
        this.incrementFrameCount();
        
        // Calculate FPS
        const instantFPS = 1000 / processingTime;
        this.fpsHistory.push(instantFPS);
        if (this.fpsHistory.length > 10) this.fpsHistory.shift();
        
        const avgFPS = this.fpsHistory.reduce((a, b) => a + b, 0) / this.fpsHistory.length;
        this.statFPS.textContent = avgFPS.toFixed(1);
        
        // Calculate average processing time
        this.processingTimes.push(processingTime);
        if (this.processingTimes.length > 10) this.processingTimes.shift();
        
        const avgTime = this.processingTimes.reduce((a, b) => a + b, 0) / this.processingTimes.length;
        this.statTime.textContent = avgTime.toFixed(0) + 'ms';
    }
    
    updateCameraStatus(status, type) {
        this.statCamera.textContent = status;
        
        const colorMap = {
            'success': '#66ff66',
            'error': '#ff6666',
            'warning': '#ffaa00',
            'inactive': '#b0b0b0'
        };
        
        this.statCamera.style.color = colorMap[type] || '#b0b0b0';
        this.statCamera.style.fontWeight = '600';
    }
    
    updateWebSocketStatus(status, text) {
        this.wsIndicator.className = 'ws-indicator ' + status;
        this.wsStatus.textContent = text;
        
        const colorMap = {
            'connected': '#66ff66',
            'disconnected': '#ff6666',
            'connecting': '#ffaa00'
        };
        
        this.wsStatus.style.color = colorMap[status] || '#b0b0b0';
        this.wsStatus.style.fontWeight = '600';
    }
    
    reset() {
        this.frameCount = 0;
        this.fpsHistory = [];
        this.processingTimes = [];
        this.lastFrameTime = 0;
        
        this.statFPS.textContent = '--';
        this.statTime.textContent = '--ms';
        this.statFrames.textContent = '0';
        this.updateCameraStatus('Inactive', 'inactive');
    }
}

