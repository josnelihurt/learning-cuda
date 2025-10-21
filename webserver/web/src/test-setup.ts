import { vi } from 'vitest';

vi.mock('./services/otel-logger', () => ({
    logger: {
        debug: vi.fn(),
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        initialize: vi.fn(),
        isDebugEnabled: vi.fn().mockReturnValue(false),
        shutdown: vi.fn(),
    },
}));
