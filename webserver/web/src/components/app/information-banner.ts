import { LitElement, html, css } from 'lit';
import { customElement } from 'lit/decorators.js';

@customElement('information-banner')
export class InformationBanner extends LitElement {
  static styles = css`
    :host {
      display: block;
      width: 100%;
      pointer-events: none;
    }

    .banner {
      background: linear-gradient(90deg, #ff6b35 0%, #f7931e 100%);
      color: white;
      padding: 4px 0;
      overflow: hidden;
      white-space: nowrap;
      font-size: 13px;
      font-weight: 500;
      border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    }

    .banner-text {
      display: inline-block;
      padding-left: 100%;
      animation: marquee 25s linear infinite;
    }

    @keyframes marquee {
      0% { transform: translate(0, 0); }
      100% { transform: translate(-100%, 0); }
    }
  `;

  render() {
    return html`
      <div class="banner">
        <div class="banner-text">
          Production deployment in progress - some components may be unavailable
        </div>
      </div>
    `;
  }
}
