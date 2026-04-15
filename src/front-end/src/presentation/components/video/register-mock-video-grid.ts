import { vi } from 'vitest';

if (!customElements.get('video-grid')) {
  class MockVideoGrid extends HTMLElement {
    applyFilterToSelected = vi.fn().mockResolvedValue(undefined);
    setManagers = vi.fn();
    getSources = vi.fn().mockReturnValue([]);
    getSelectedSourceIds = vi.fn().mockReturnValue(new Set<string>());
    addSource = vi.fn().mockReturnValue(true);
    changeSourceImage = vi.fn();
  }
  customElements.define('video-grid', MockVideoGrid);
}
