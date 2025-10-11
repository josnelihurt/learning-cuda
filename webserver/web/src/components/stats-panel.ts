import { LitElement, html, css } from 'lit';
import { customElement, property } from 'lit/decorators.js';

export type CameraStatusType = 'success' | 'error' | 'warning' | 'inactive';
export type WebSocketStatusType = 'connected' | 'disconnected' | 'connecting';

@customElement('stats-panel')
export class StatsPanel extends LitElement {
    @property({ type: String }) fps = '--';
    @property({ type: String }) time = '--ms';
    @property({ type: Number }) frames = 0;
    @property({ type: String }) cameraStatus = 'Inactive';
    @property({ type: String }) cameraStatusType: CameraStatusType = 'inactive';
    @property({ type: String }) wsStatus = 'Connecting...';
    @property({ type: String }) wsStatusType: WebSocketStatusType = 'connecting';

    private frameCount = 0;
    private fpsHistory: number[] = [];
    private processingTimes: number[] = [];
    private lastFrameTime = 0;

    static styles = css`
        :host {
            grid-area: stats;
            display: block;
            background: #2a2a2a;
            color: white;
            border-top: 2px solid #404040;
            padding: 0 var(--spacing-lg, 30px);
            z-index: var(--z-stats, 1000);
        }

        .stats-container {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: var(--spacing-xl, 40px);
            flex-wrap: wrap;
        }

        .stat-item {
            display: flex;
            align-items: center;
            gap: var(--spacing-xs, 8px);
            font-size: 14px;
            color: #e0e0e0;
        }

        .stat-label {
            color: #b0b0b0;
            font-weight: 500;
        }

        strong {
            font-weight: 600;
            font-size: 16px;
            min-width: 60px;
            text-align: left;
        }

        .camera-status {
            font-weight: 600;
        }

        .camera-status.success {
            color: #66ff66;
        }

        .camera-status.error {
            color: #ff6666;
        }

        .camera-status.warning {
            color: #ffaa00;
        }

        .camera-status.inactive {
            color: #b0b0b0;
        }

        .ws-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 4px;
        }

        .ws-indicator.connected {
            background: #66ff66;
            box-shadow: 0 0 8px #66ff66;
        }

        .ws-indicator.disconnected {
            background: #ff6666;
            box-shadow: 0 0 8px #ff6666;
        }

        .ws-indicator.connecting {
            background: #ffaa00;
            box-shadow: 0 0 8px #ffaa00;
            animation: pulse 1.5s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .ws-status {
            font-weight: 600;
        }

        .ws-status.connected {
            color: #66ff66;
        }

        .ws-status.disconnected {
            color: #ff6666;
        }

        .ws-status.connecting {
            color: #ffaa00;
        }
    `;

    render() {
        return html`
            <div class="stats-container">
                <div class="stat-item">
                    <span class="stat-label">FPS:</span>
                    <strong>${this.fps}</strong>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Time:</span>
                    <strong>${this.time}</strong>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Frames:</span>
                    <strong>${this.frames}</strong>
                </div>
                <div class="stat-item">
                    <span class="camera-status ${this.cameraStatusType}">
                        ${this.cameraStatus}
                    </span>
                </div>
                <div class="stat-item">
                    <span class="ws-indicator ${this.wsStatusType}"></span>
                    <span class="ws-status ${this.wsStatusType}">${this.wsStatus}</span>
                </div>
            </div>
        `;
    }

    incrementFrameCount(): void {
        this.frameCount++;
        this.frames = this.frameCount;
    }

    updateProcessingStats(processingTime: number): void {
        this.incrementFrameCount();

        const instantFPS = 1000 / processingTime;
        this.fpsHistory.push(instantFPS);
        if (this.fpsHistory.length > 10) this.fpsHistory.shift();

        const avgFPS = this.fpsHistory.reduce((a, b) => a + b, 0) / this.fpsHistory.length;
        this.fps = avgFPS.toFixed(1);

        this.processingTimes.push(processingTime);
        if (this.processingTimes.length > 10) this.processingTimes.shift();

        const avgTime = this.processingTimes.reduce((a, b) => a + b, 0) / this.processingTimes.length;
        this.time = avgTime.toFixed(0) + 'ms';
    }

    updateCameraStatus(status: string, type: CameraStatusType): void {
        this.cameraStatus = status;
        this.cameraStatusType = type;
    }

    updateWebSocketStatus(status: WebSocketStatusType, text: string): void {
        this.wsStatusType = status;
        this.wsStatus = text;
    }

    reset(): void {
        this.frameCount = 0;
        this.fpsHistory = [];
        this.processingTimes = [];
        this.lastFrameTime = 0;

        this.fps = '--';
        this.time = '--ms';
        this.frames = 0;
        this.updateCameraStatus('Inactive', 'inactive');
    }
}

declare global {
    interface HTMLElementTagNameMap {
        'stats-panel': StatsPanel;
    }
}

