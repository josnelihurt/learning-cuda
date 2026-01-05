import { LitElement, html, css } from 'lit';
import { customElement, state, property } from 'lit/decorators.js';
import type { ToolCategory } from '../../gen/config_service_pb';

@customElement('tools-dropdown')
export class ToolsDropdown extends LitElement {
  @state() private isOpen = false;
  @property({ type: Array }) categories: ToolCategory[] = [];

  static styles = css`
    :host {
      position: relative;
      display: inline-block;
    }

    .dropdown-trigger {
      padding: 6px 16px;
      border-radius: 8px;
      border: 2px solid var(--border-color, rgba(255, 255, 255, 0.15));
      background: var(--background, rgba(20, 20, 30, 0.8));
      color: var(--text-primary, white);
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .dropdown-trigger:hover {
      border-color: var(--primary-color, #00d9ff);
      color: var(--primary-color, #00d9ff);
      transform: scale(1.05);
    }

    .dropdown-trigger.open {
      border-color: var(--primary-color, #00d9ff);
      color: var(--primary-color, #00d9ff);
    }

    .arrow {
      font-size: 10px;
      transition: transform 0.2s ease;
    }

    .arrow.open {
      transform: rotate(180deg);
    }

    .dropdown-menu {
      position: absolute;
      top: calc(100% + 8px);
      right: 0;
      background: rgba(30, 30, 40, 0.98);
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 8px;
      min-width: 200px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      z-index: 1000;
      overflow: hidden;
      opacity: 0;
      transform: translateY(-10px);
      pointer-events: none;
      transition: all 0.2s ease;
    }

    .dropdown-menu.open {
      opacity: 1;
      transform: translateY(0);
      pointer-events: all;
    }

    .dropdown-item {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 12px 16px;
      color: rgba(255, 255, 255, 0.9);
      text-decoration: none;
      transition: all 0.2s ease;
      cursor: pointer;
      border-left: 3px solid transparent;
    }

    .dropdown-item:hover {
      background: rgba(255, 255, 255, 0.1);
      border-left-color: var(--accent-color, #00d9ff);
      color: white;
    }

    .dropdown-item-icon {
      font-size: 16px;
      width: 20px;
      height: 20px;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .dropdown-item-icon img {
      width: 16px;
      height: 16px;
      object-fit: contain;
    }

    .dropdown-item-text {
      flex: 1;
    }

    .dropdown-divider {
      height: 1px;
      background: rgba(255, 255, 255, 0.1);
      margin: 4px 0;
    }
  `;

  connectedCallback() {
    super.connectedCallback();
    document.addEventListener('click', this.handleClickOutside);
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    document.removeEventListener('click', this.handleClickOutside);
  }

  private handleClickOutside = (e: Event) => {
    if (!this.contains(e.target as Node)) {
      this.isOpen = false;
    }
  };

  private toggleDropdown() {
    this.isOpen = !this.isOpen;
  }

  private handleToolClick(tool: any) {
    if (tool.type === 'url' && tool.url) {
      window.open(tool.url, '_blank');
    } else if (tool.type === 'action') {
      this.executeAction(tool.action);
    }
    this.isOpen = false;
  }

  private executeAction(action: string) {
    if (action === 'sync_flags') {
      const syncButton = document.querySelector('sync-flags-button');
      if (syncButton) {
        const button = syncButton.shadowRoot?.querySelector('button');
        button?.click();
      }
    }
  }

  private renderTool(tool: any) {
    const iconContent = tool.iconPath
      ? html`<img src="${tool.iconPath}" alt="${tool.name}" />`
      : html`${tool.type === 'action' && tool.action === 'sync_flags' ? 'R' : 'OK'}`;

    return html`
      <a
        class="dropdown-item"
        @click=${() => this.handleToolClick(tool)}
        data-testid="tool-item-${tool.name.toLowerCase().replace(/\s+/g, '-')}"
      >
        <span class="dropdown-item-icon">${iconContent}</span>
        <span class="dropdown-item-text">${tool.name}</span>
      </a>
    `;
  }

  render() {
    return html`
      <button
        class="dropdown-trigger ${this.isOpen ? 'open' : ''}"
        @click=${this.toggleDropdown}
        data-testid="tools-dropdown-button"
      >
        <span>Tools</span>
        <span class="arrow ${this.isOpen ? 'open' : ''}">▼</span>
      </button>

      <div class="dropdown-menu ${this.isOpen ? 'open' : ''}" data-testid="tools-dropdown-menu">
        ${this.categories.map(
          (category, index) => html`
            ${category.tools.map((tool) => this.renderTool(tool))}
            ${index < this.categories.length - 1 ? html`<div class="dropdown-divider"></div>` : ''}
          `
        )}
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'tools-dropdown': ToolsDropdown;
  }
}
