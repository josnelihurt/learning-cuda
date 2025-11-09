import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { styleMap } from 'lit/directives/style-map.js';

type TourStep = {
  id: string;
  selector: string;
  title: string;
  description: string;
};

const DISMISS_KEY = 'cuda-app-tour-dismissed';

@customElement('app-tour')
export class AppTour extends LitElement {
  static styles = css`
    :host {
      display: block;
    }

    :host([hidden]) {
      display: none;
    }

    .overlay {
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.45);
      backdrop-filter: blur(1.5px);
      z-index: 2200;
      pointer-events: auto;
    }

    .overlay.hidden {
      pointer-events: none;
    }

    .focus-ring {
      position: fixed;
      border-radius: 14px;
      box-shadow: 0 0 0 2000px rgba(15, 23, 42, 0.45);
      border: 2px solid var(--accent-color, #38bdf8);
      background: rgba(56, 189, 248, 0.08);
      pointer-events: none;
      transition: all 0.22s ease;
    }

    .content {
      position: fixed;
      max-width: 320px;
      color: #f8fafc;
      background: rgba(15, 23, 42, 0.88);
      border-radius: 16px;
      padding: 20px 24px;
      box-shadow: 0 18px 42px rgba(15, 23, 42, 0.45);
      display: flex;
      flex-direction: column;
      gap: 16px;
      transition: transform 0.2s ease, opacity 0.2s ease;
    }

    .content::after {
      content: '';
      position: absolute;
      width: 12px;
      height: 12px;
      background: inherit;
      transform: rotate(45deg);
      z-index: -1;
    }

    .content.placement-right::after {
      left: -6px;
      top: 50%;
      margin-top: -6px;
      box-shadow: -3px 3px 10px rgba(15, 23, 42, 0.35);
    }

    .content.placement-left::after {
      right: -6px;
      top: 50%;
      margin-top: -6px;
      box-shadow: 3px -3px 10px rgba(15, 23, 42, 0.35);
    }

    .content.placement-bottom::after {
      top: -6px;
      left: 50%;
      margin-left: -6px;
      box-shadow: -3px -3px 10px rgba(15, 23, 42, 0.35);
    }

    .content.placement-top::after {
      bottom: -6px;
      left: 50%;
      margin-left: -6px;
      box-shadow: 3px 3px 10px rgba(15, 23, 42, 0.35);
    }

    .step-label {
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: rgba(148, 163, 184, 0.9);
    }

    .step-title {
      font-size: 20px;
      font-weight: 700;
      margin: 0;
    }

    .step-desc {
      font-size: 15px;
      line-height: 1.5;
      margin: 0;
      color: rgba(226, 232, 240, 0.92);
    }

    .actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }

    button {
      border: none;
      border-radius: 999px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.15s ease, opacity 0.2s ease;
    }

    button.primary {
      padding: 10px 20px;
      background: #38bdf8;
      color: #0f172a;
    }

    button.primary:hover {
      transform: translateY(-1px);
    }

    button.secondary {
      padding: 8px 16px;
      background: transparent;
      color: rgba(226, 232, 240, 0.92);
    }

    button.secondary:hover {
      opacity: 0.8;
    }

    @media (max-width: 640px) {
      .content {
        max-width: calc(100vw - 32px);
        padding: 16px 20px;
      }
    }
  `;

  @state() private active = false;
  @state() private stepIndex = 0;
  @state() private focusStyle: Record<string, string> = {};
  @state() private tooltipStyle: Record<string, string> = {};
  @state() private tooltipPlacement: 'right' | 'left' | 'top' | 'bottom' = 'right';

  private readonly steps: TourStep[] = [
    {
      id: 'add-source',
      selector: 'button[data-testid="add-input-fab"], add-source-fab',
      title: 'Add Input',
      description:
        'Use Add Input to bring new static images, live cameras, or videos into the workspace.',
    },
    {
      id: 'filter-panel',
      selector: '.sidebar .filters-section, filter-panel',
      title: 'Filter Panel',
      description:
        'Drag, toggle, and fine tune filters. GPU runs filters on server GPU, while CPU keeps everything on server CPU processing.',
    },
    {
      id: 'change-image',
      selector: 'video-source-card[data-source-number="1"], video-grid',
      title: 'Switch Images',
      description:
        'Select a source and use the change image control (right upper corner) to swap between available test images.',
    },
    {
      id: 'tools-dropdown',
      selector: '[data-testid="tools-dropdown-button"], tools-dropdown',
      title: 'Tools Menu',
      description: 'Open Tools to access Grafana, Jaeger, Playwright reports, and other utilities.',
    },
    {
      id: 'feature-flags',
      selector: 'feature-flags-button',
      title: 'Feature Flags',
      description: 'Manage feature toggles for experiments. Connect to Flipt to enable or disable flags.',
    },
    {
      id: 'version-info',
      selector: 'version-tooltip-lit button.info-btn, version-tooltip-lit',
      title: 'Version Details',
      description: 'Click the info icon to see build details for the frontend, backend, CPP library and more.',
    },
  ];

  private currentTarget: HTMLElement | null = null;
  private resizeHandler = () => this.updateLayout();
  private animationFrame = 0;

  disconnectedCallback(): void {
    super.disconnectedCallback();
    window.removeEventListener('resize', this.resizeHandler);
    window.removeEventListener('scroll', this.resizeHandler, true);
    cancelAnimationFrame(this.animationFrame);
    this.clearHighlight();
  }

  render() {
    if (!this.active) {
      return html``;
    }

    const step = this.steps[this.stepIndex];
    const total = this.steps.length;

    return html`
      <div class="overlay" role="presentation">
        <div class="focus-ring" style=${styleMap(this.focusStyle)}></div>
        <div
          class="content placement-${this.tooltipPlacement}"
          style=${styleMap(this.tooltipStyle)}
          role="dialog"
          aria-modal="true"
          aria-labelledby="${step.id}-title"
        >
          <span class="step-label">Step ${this.stepIndex + 1} of ${total}</span>
          <h2 id="${step.id}-title" class="step-title">${step.title}</h2>
          <p class="step-desc">${step.description}</p>
          <div class="actions">
            <button class="secondary" @click=${this.skip}>Skip</button>
            <button class="primary" @click=${this.next}>
              ${this.stepIndex === total - 1 ? 'Got it' : 'Next'}
            </button>
          </div>
        </div>
      </div>
    `;
  }

  public startIfNeeded(): void {
    if (this.isDismissed()) return;
    this.start();
  }

  public start(force = false): void {
    if (!force && this.isDismissed()) {
      return;
    }
    if (this.active) {
      this.stepIndex = 0;
      this.updateLayout(true);
      return;
    }
    this.active = true;
    this.stepIndex = 0;
    this.attachListeners();
    this.updateLayout(true);
  }

  public resetForTesting(): void {
    localStorage.removeItem(DISMISS_KEY);
    this.active = false;
    this.stepIndex = 0;
    this.focusStyle = {};
    this.tooltipStyle = {};
    this.clearHighlight();
  }

  private next(): void {
    if (this.stepIndex >= this.steps.length - 1) {
      this.complete();
      return;
    }
    this.stepIndex += 1;
    this.updateLayout(true);
  }

  private skip(): void {
    this.dismiss();
    this.deactivate();
  }

  private complete(): void {
    this.dismiss();
    this.deactivate();
  }

  private deactivate(): void {
    this.active = false;
    window.removeEventListener('resize', this.resizeHandler);
    window.removeEventListener('scroll', this.resizeHandler, true);
    cancelAnimationFrame(this.animationFrame);
    this.clearHighlight();
  }

  private dismiss(): void {
    try {
      localStorage.setItem(DISMISS_KEY, 'true');
    } catch {
      // ignore storage errors
    }
  }

  private isDismissed(): boolean {
    try {
      return localStorage.getItem(DISMISS_KEY) === 'true';
    } catch {
      return false;
    }
  }

  private attachListeners(): void {
    window.addEventListener('resize', this.resizeHandler);
    window.addEventListener('scroll', this.resizeHandler, true);
  }

  private updateLayout(withScroll = false): void {
    const step = this.steps[this.stepIndex];
    if (!step) return;

    const target = this.findTarget(step.selector);
    if (!target) {
      if (!withScroll && this.focusStyle.width) {
        this.scheduleRetry();
        return;
      }

      const fallbackRoot = this.findTarget('video-grid, filter-panel, add-source-fab, body');
      const bounds = fallbackRoot?.getBoundingClientRect();
      const fallbackRect = bounds
        ? bounds
        : new DOMRect(0, 0, window.innerWidth, window.innerHeight);

      this.focusStyle = {
        top: `${fallbackRect.top}px`,
        left: `${fallbackRect.left}px`,
        width: `${fallbackRect.width}px`,
        height: `${fallbackRect.height}px`,
      };
      const { style, placement } = this.calculateTooltipPosition(fallbackRect);
      this.tooltipStyle = style;
      this.tooltipPlacement = placement;
      return;
    }

    this.currentTarget = target;

    if (withScroll) {
      target.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }

    const rect = target.getBoundingClientRect();
    const padding = 16;

    this.focusStyle = {
      top: `${Math.max(rect.top - padding, 16)}px`,
      left: `${Math.max(rect.left - padding, 16)}px`,
      width: `${rect.width + padding * 2}px`,
      height: `${rect.height + padding * 2}px`,
    };

    const { style, placement } = this.calculateTooltipPosition(rect);
    this.tooltipStyle = style;
    this.tooltipPlacement = placement;
  }

  private calculateTooltipPosition(
    rect: DOMRect
  ): { style: Record<string, string>; placement: 'right' | 'left' | 'top' | 'bottom' } {
    const margin = 20;
    const tooltipWidth = 320;
    const tooltipHeight = 220;
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const centerY = rect.top + rect.height / 2;
    const centerX = rect.left + rect.width / 2;

    const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

    if (rect.right + margin + tooltipWidth <= viewportWidth) {
      const top = clamp(centerY - tooltipHeight / 2, margin, viewportHeight - tooltipHeight - margin);
      return {
        style: {
          top: `${top}px`,
          left: `${rect.right + margin}px`,
        },
        placement: 'right',
      };
    }

    if (rect.left - margin - tooltipWidth >= 0) {
      const top = clamp(centerY - tooltipHeight / 2, margin, viewportHeight - tooltipHeight - margin);
      return {
        style: {
          top: `${top}px`,
          left: `${rect.left - tooltipWidth - margin}px`,
        },
        placement: 'left',
      };
    }

    if (rect.bottom + margin + tooltipHeight <= viewportHeight) {
      const left = clamp(centerX - tooltipWidth / 2, margin, viewportWidth - tooltipWidth - margin);
      return {
        style: {
          top: `${rect.bottom + margin}px`,
          left: `${left}px`,
        },
        placement: 'bottom',
      };
    }

    const left = clamp(centerX - tooltipWidth / 2, margin, viewportWidth - tooltipWidth - margin);
    const top = clamp(rect.top - tooltipHeight - margin, margin, viewportHeight - tooltipHeight - margin);
    return {
      style: {
        top: `${top}px`,
        left: `${left}px`,
      },
      placement: 'top',
    };
  }

  private scheduleRetry(): void {
    cancelAnimationFrame(this.animationFrame);
    this.animationFrame = requestAnimationFrame(() => this.updateLayout());
  }

  private clearHighlight(): void {
    this.currentTarget = null;
  }

  private findTarget(selector: string): HTMLElement | null {
    const direct = document.querySelector(selector);
    if (direct instanceof HTMLElement) {
      return direct;
    }

    const body = document.body;
    if (!body) {
      return null;
    }

    const stack: (Element | ShadowRoot)[] = [body];
    const visited = new Set<Element | ShadowRoot>();

    while (stack.length > 0) {
      const node = stack.pop()!;
      if (visited.has(node)) {
        continue;
      }
      visited.add(node);

      if (node instanceof HTMLElement && node.matches(selector)) {
        return node;
      }

      const children = Array.from(node.children ?? []);
      for (let i = children.length - 1; i >= 0; i -= 1) {
        stack.push(children[i]);
      }

      if (node instanceof Element && node.shadowRoot) {
        stack.push(node.shadowRoot);
      }
    }

    return null;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'app-tour': AppTour;
  }
}

