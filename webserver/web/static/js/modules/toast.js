/**
 * ToastManager - Modern toast notification system
 */
const toastTiemout = 7000;
class ToastManager {
    constructor(config = {}) {
        this.config = {
            duration: config.duration || toastTiemout, // 7 seconds default
            maxToasts: config.maxToasts || 5,
            position: config.position || 'top-right'
        };
        
        this.toasts = [];
        this.container = null;
        this.init();
    }
    
    init() {
        // Create toast container
        this.container = document.createElement('div');
        this.container.className = 'toast-container';
        this.container.id = 'toastContainer';
        document.body.appendChild(this.container);
    }
    
    /**
     * Show a toast notification
     * @param {string} type - 'success', 'error', 'warning', 'info'
     * @param {string} title - Toast title
     * @param {string} message - Toast message (optional)
     * @param {number} duration - Custom duration in ms (optional)
     */
    show(type, title, message = '', duration = null) {
        // Limit number of toasts
        if (this.toasts.length >= this.config.maxToasts) {
            this.dismiss(this.toasts[0]);
        }
        
        const toast = this.createToast(type, title, message);
        this.container.appendChild(toast);
        this.toasts.push(toast);
        
        // Trigger animation
        setTimeout(() => toast.classList.add('show'), 10);
        
        // Auto-dismiss
        const dismissDuration = duration !== null ? duration : this.config.duration;
        if (dismissDuration > 0) {
            setTimeout(() => this.dismiss(toast), dismissDuration);
        }
        
        return toast;
    }
    
    createToast(type, title, message) {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        const iconMap = {
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️'
        };
        
        toast.innerHTML = `
            <div class="toast-icon">${iconMap[type] || 'ℹ️'}</div>
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                ${message ? `<div class="toast-message">${message}</div>` : ''}
            </div>
            <button class="toast-close" onclick="window.app.toastManager.dismiss(this.parentElement)">×</button>
        `;
        
        return toast;
    }
    
    dismiss(toast) {
        if (!toast || !toast.parentElement) return;
        
        toast.classList.remove('show');
        toast.classList.add('hide');
        
        setTimeout(() => {
            if (toast.parentElement) {
                toast.parentElement.removeChild(toast);
            }
            const index = this.toasts.indexOf(toast);
            if (index > -1) {
                this.toasts.splice(index, 1);
            }
        }, 300);
    }
    
    dismissAll() {
        this.toasts.forEach(toast => this.dismiss(toast));
    }
    
    // Convenience methods
    success(title, message, duration) {
        return this.show('success', title, message, duration);
    }
    
    error(title, message, duration) {
        return this.show('error', title, message, duration);
    }
    
    warning(title, message, duration) {
        return this.show('warning', title, message, duration);
    }
    
    info(title, message, duration) {
        return this.show('info', title, message, duration);
    }
    
    // Update configuration
    setDuration(duration) {
        this.config.duration = duration;
    }
}

