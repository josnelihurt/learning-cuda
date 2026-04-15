import { describe, expect, it } from 'vitest';
import { WebRTCService } from '@/infrastructure/connection/webrtc-service';

describe('WebRTCService with test-setup stubs', () => {
  it('constructs without throwing when globals are stubbed', () => {
    expect(() => new WebRTCService()).not.toThrow();
  });
});
