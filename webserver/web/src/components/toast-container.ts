import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { repeat } from 'lit/directives/repeat.js';

export type ToastType = 'success' | 'error' | 'warning' | 'info';

interface ToastConfig {
    duration?: number;
    maxToasts?: number;
    skipAnimations?: boolean;
}

export interface Toast {
    id: string;
    type: ToastType;
    title: string;
    message: string;
    show: boolean;
}

const DEFAULT_DURATION = 7000;
const MAX_TOASTS = 5;

@customElement('toast-container')
export class ToastContainer extends LitElement {
    @state() private toasts: Toast[] = [];
    
    private config: Required<ToastConfig> = {
        duration: DEFAULT_DURATION,
        maxToasts: MAX_TOASTS,
        skipAnimations: false
    };
    
    static styles = css`
        :host {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            display: flex;
            flex-direction: column;
            gap: 12px;
            pointer-events: none;
        }
        
        .toast {
            pointer-events: auto;
            display: flex;
            align-items: flex-start;
            gap: 12px;
            padding: 16px 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            min-width: 300px;
            max-width: 400px;
            opacity: 0;
            transform: translateX(100%);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .toast.show {
            opacity: 1;
            transform: translateX(0);
        }
        
        .toast.hide {
            opacity: 0;
            transform: translateX(100%);
        }
        
        .toast-icon {
            font-size: 24px;
            line-height: 1;
            flex-shrink: 0;
        }
        
        .toast-content {
            flex: 1;
            min-width: 0;
        }
        
        .toast-title {
            font-weight: 600;
            font-size: 15px;
            margin-bottom: 4px;
            color: #1a1a1a;
        }
        
        .toast-message {
            font-size: 14px;
            color: #666;
            line-height: 1.4;
        }
        
        .toast-close {
            background: none;
            border: none;
            font-size: 24px;
            line-height: 1;
            cursor: pointer;
            color: #999;
            padding: 0;
            margin: -4px -4px 0 0;
            flex-shrink: 0;
            transition: color 0.2s;
        }
        
        .toast-close:hover {
            color: #333;
        }
        
        .toast-success {
            border-left: 4px solid #4caf50;
        }
        
        .toast-error {
            border-left: 4px solid #f44336;
        }
        
        .toast-warning {
            border-left: 4px solid #ff9800;
        }
        
        .toast-info {
            border-left: 4px solid #2196f3;
        }
    `;

    render() {
        return html`
            ${repeat(this.toasts, (toast) => toast.id, (toast) => html`
                <div 
                    class="toast toast-${toast.type} ${toast.show ? 'show' : ''}"
                    @animationend=${() => this.handleAnimationEnd(toast.id)}
                >
                    <div class="toast-icon">${this.getIcon(toast.type)}</div>
                    <div class="toast-content">
                        <div class="toast-title">${toast.title}</div>
                        ${toast.message ? html`<div class="toast-message">${toast.message}</div>` : ''}
                    </div>
                    <button 
                        class="toast-close" 
                        @click=${() => this.dismiss(toast.id)}
                        aria-label="Close"
                    >×</button>
                </div>
            `)}
        `;
    }

    configure(config: ToastConfig): void {
        this.config = {
            ...this.config,
            ...config
        };
    }

    show(type: ToastType, title: string, message = '', duration: number | null = null): string {
        if (this.toasts.length >= this.config.maxToasts) {
            this.dismiss(this.toasts[0].id);
        }

        const id = `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        const toast: Toast = {
            id,
            type,
            title,
            message,
            show: false
        };

        this.toasts = [...this.toasts, toast];
        this.requestUpdate();

        if (this.config.skipAnimations) {
            setTimeout(() => {
                this.toasts = this.toasts.map(t => 
                    t.id === id ? { ...t, show: true } : t
                );
                this.requestUpdate();
            }, 0);
        } else {
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    this.toasts = this.toasts.map(t => 
                        t.id === id ? { ...t, show: true } : t
                    );
                });
            });
        }

        const dismissDuration = duration !== null ? duration : this.config.duration;
        if (dismissDuration > 0) {
            setTimeout(() => this.dismiss(id), dismissDuration);
        }

        return id;
    }

    dismiss(id: string): void {
        const toastIndex = this.toasts.findIndex(t => t.id === id);
        if (toastIndex === -1) return;

        this.toasts = this.toasts.map(t => 
            t.id === id ? { ...t, show: false } : t
        );

        setTimeout(() => {
            this.toasts = this.toasts.filter(t => t.id !== id);
        }, 300);
    }

    dismissAll(): void {
        this.toasts.forEach(toast => this.dismiss(toast.id));
    }

    success(title: string, message = '', duration: number | null = null): string {
        return this.show('success', title, message, duration);
    }

    error(title: string, message = '', duration: number | null = null): string {
        return this.show('error', title, message, duration);
    }

    warning(title: string, message = '', duration: number | null = null): string {
        return this.show('warning', title, message, duration);
    }

    info(title: string, message = '', duration: number | null = null): string {
        return this.show('info', title, message, duration);
    }

    setDuration(duration: number): void {
        this.config.duration = duration;
    }

    getToastCount(): number {
        return this.toasts.length;
    }

    getToasts(): Toast[] {
        return this.toasts;
    }

    private getIcon(type: ToastType): string {
        const iconMap: Record<ToastType, string> = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };
        return iconMap[type] || 'ℹ️';
    }

    private handleAnimationEnd(id: string): void {
        const toast = this.toasts.find(t => t.id === id);
        if (toast && !toast.show) {
            this.toasts = this.toasts.filter(t => t.id !== id);
        }
    }
}

declare global {
    interface HTMLElementTagNameMap {
        'toast-container': ToastContainer;
    }
}

