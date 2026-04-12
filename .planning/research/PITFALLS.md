# Pitfalls Research

**Domain:** Lit Web Components to React migration — CUDA image/video processing platform with WebRTC, ConnectRPC, Vite
**Researched:** 2026-04-12
**Confidence:** HIGH (Lit/React interop, WebRTC), MEDIUM (Vite MPA specifics, ConnectRPC React patterns)

---

## Critical Pitfalls

### Pitfall 1: srcObject Cannot Be Set as JSX Prop — Silent Failure

**What goes wrong:**
React does not recognize `srcObject` as a valid DOM prop. Writing `<video srcObject={stream} />` either silently ignores the assignment or causes a React warning. The video element renders with no source, appearing as a blank black box. This is the most common first blocker when porting WebRTC video display from Lit (which uses direct property binding via `.srcObject=${stream}`) to React.

**Why it happens:**
React maps JSX props to HTML attributes, not DOM properties. `srcObject` is a DOM property with no corresponding HTML attribute. Lit's `.property` binding syntax sets DOM properties directly, so the Lit pattern works. Developers copy the concept without recognizing the mechanism difference.

**How to avoid:**
Always set `srcObject` imperatively via `useRef` and `useEffect`:
```typescript
const videoRef = useRef<HTMLVideoElement>(null);
useEffect(() => {
  if (videoRef.current && stream) {
    videoRef.current.srcObject = stream;
  }
}, [stream]);
return <video ref={videoRef} autoPlay playsInline muted />;
```
Never pass `srcObject` as a JSX prop. This is not a workaround — it is the correct React pattern.

**Warning signs:**
Video element renders with no content. No console error (React silently drops unrecognized props in production builds). The Lit version works but the React version shows a blank video.

**Phase to address:**
Phase that implements video streaming / WebRTC display (any component rendering a `<video>` element with a live stream).

---

### Pitfall 2: Shadow DOM Event Bubbling Breaks React's Synthetic Event System

**What goes wrong:**
Lit components use Shadow DOM by default. Events dispatched from inside a shadow root undergo "event retargeting" at the shadow boundary — `event.target` changes to the host element. React's synthetic event system delegates all events to the root container, which means events fired inside shadow DOM may not reach React's handler, or reach it with the wrong target. A confirmed React bug (issue #24136) causes events originating inside a shadow root to fire multiple times as they bubble up the virtual DOM tree.

**Why it happens:**
React's event delegation model assumes a flat DOM. Shadow DOM creates isolated event scopes. This conflict is inherent and not fixed in React 18 for all event types.

**How to avoid:**
During the coexistence period (both Lit and React frontends active), ensure the Lit frontend is served from a completely separate HTML entry point and DOM root, not embedded as children of a React component tree. Do not attempt to render Lit custom elements inside React JSX as a bridge strategy. The `/lit` route and `/react` route must have independent DOM roots with no shared parent element.

If Lit components must emit events that React needs to catch, use the `ref` callback pattern with `addEventListener` directly instead of React's `onXxx` JSX props.

**Warning signs:**
Event handlers fire zero times or multiple times. `event.target` is a host element instead of the originating element. Issues appear only when Lit and React components are in the same DOM subtree.

**Phase to address:**
Phase 1 (scaffold) — must establish routing architecture before any component work begins.

---

### Pitfall 3: Vite Single Entry Point Config Breaks with Multiple HTML Files

**What goes wrong:**
The current `vite.config.ts` has a single entry: `input: resolve(__dirname, 'src/main.ts')`. Naively adding a second React entry by passing an object to `rollupOptions.input` causes the dev server to return 404 for routes that don't have a corresponding HTML file at the exact URL path. HTML5 history routing (`/react/*` deep routes) returns 404 because Vite's dev server only serves HTML for files it knows about.

**Why it happens:**
Vite's MPA (multi-page app) mode requires each page to have its own `index.html` at the directory level that maps to the URL. Vite ignores the key name in `rollupOptions.input` for HTML files — it uses the resolved file path to determine the output URL. The existing Vite config also has a custom gRPC proxy and HTTPS configuration that must be preserved exactly.

**How to avoid:**
Structure the MPA correctly before writing a single React component:
1. Create `webserver/web/react/index.html` (separate HTML file)
2. Create `webserver/web/react/main.tsx` (React entry point)
3. Update `rollupOptions.input` to include both entries:
   ```typescript
   input: {
     lit: resolve(__dirname, 'src/main.ts'),
     react: resolve(__dirname, 'react/index.html'),
   }
   ```
4. Add a dev server fallback or configure the Go server to serve the correct HTML for `/react/*` routes
5. Preserve all existing proxy config, HTTPS config, and HMR config — do not regenerate vite.config.ts from scratch

The existing gRPC proxy configuration (`/grpc` rewrite with content-type header manipulation) must be preserved verbatim — it handles grpc-web protocol translation that ConnectRPC requires.

**Warning signs:**
`404 Not Found` when navigating to `/react` in dev. React app loads but gRPC calls fail (proxy not configured). HMR stops working for the Lit frontend after config change.

**Phase to address:**
Phase 1 (scaffold) — first task before any component work.

---

### Pitfall 4: React Context with High-Frequency State Causes Render Storms During Video Processing

**What goes wrong:**
Placing high-frequency state (frame counters, processing latency metrics, WebSocket frame data, WebRTC stats) into a React Context causes all context consumers to re-render on every update. At 30fps video, this means 30 re-renders per second for every component that consumes the context — including filter controls, file lists, and settings UI that have nothing to do with the video frame. The result is janky UI, dropped frames in the processed video display, and CPU pegged at high utilization.

**Why it happens:**
React Context re-renders every subscriber when the context value object changes reference — even if the consuming component only reads a small slice. The Lit version uses reactive properties scoped to individual elements with no equivalent global re-render cascade. Developers migrating from Lit naturally reach for a single "app state" context to replace Lit's component-level reactivity.

**How to avoid:**
Split context by update frequency. Use separate contexts for:
- **Static/slow state**: user settings, filter configuration, file list — fine in Context
- **High-frequency state**: current frame data, processing stats, stream status — use `useRef` (not state) for values that drive imperative updates, or Zustand with subscriptions that bypass React's render cycle

Never store a `MediaStream`, `RTCPeerConnection`, or frame buffer in React state or Context. Store them in `useRef` — they are mutable objects that should not trigger re-renders when they change.

**Warning signs:**
React DevTools Profiler shows components re-rendering that have no visual change. Frame display stutters when filter controls are visible. CPU usage spikes when moving the mouse over any UI element while streaming.

**Phase to address:**
Phase implementing video streaming hook (`useVideoStream`) — establish the pattern before building consumer components.

---

### Pitfall 5: WebRTC Resource Leaks from Missing useEffect Cleanup

**What goes wrong:**
`getUserMedia` streams, `RTCPeerConnection` instances, and `MediaStreamTrack` objects continue running after the React component unmounts. Camera LED stays on after navigating away. Multiple peer connections accumulate across route changes. Memory grows until the browser tab crashes or camera access is permanently held.

**Why it happens:**
In Lit, cleanup lives in `disconnectedCallback()` which fires reliably when the element is removed from DOM. React's `useEffect` cleanup function is the equivalent, but it only runs if explicitly returned. React 18 Strict Mode mounts components twice in development, causing double-initialization of WebRTC resources with only one cleanup — exposing bugs that production (single mount) hides.

**How to avoid:**
Every `useEffect` that initializes WebRTC resources must return a cleanup function:
```typescript
useEffect(() => {
  let stream: MediaStream | null = null;
  navigator.mediaDevices.getUserMedia({ video: true }).then(s => {
    stream = s;
    // assign to ref, not state
  });
  return () => {
    stream?.getTracks().forEach(track => track.stop());
    peerConnectionRef.current?.close();
    peerConnectionRef.current = null;
  };
}, []);
```
Test cleanup by navigating away and back. Verify camera LED turns off. Run with React Strict Mode enabled in development — if initialization fails on double-mount, the cleanup is incomplete.

**Warning signs:**
Camera LED stays on after navigating away. Browser console shows "getUserMedia called while already active." Memory usage grows monotonically during navigation. Multiple `RTCPeerConnection` instances visible in `chrome://webrtc-internals`.

**Phase to address:**
Phase implementing `useVideoStream` hook — design cleanup before first usage.

---

### Pitfall 6: DIContainer Singleton Is Not React-Compatible

**What goes wrong:**
The existing `DIContainer.getInstance()` singleton pattern (documented with a TODO in `webserver/web/src/application/di/Container.ts`) creates services once at module load time. In React, this means services are shared across the entire app lifecycle including across hot reloads in development, cannot be reset between tests, and cannot be provided via React Context with different configurations. Tests that rely on mocking services cannot isolate the singleton state between test cases, causing test pollution and order-dependent failures.

**Why it happens:**
The singleton was designed for Lit where the DI container is initialized once per page load and components access it imperatively. React's component model expects dependencies to flow down via props or context — singletons created outside the React tree are invisible to React's lifecycle.

**How to avoid:**
Do not import `DIContainer.getInstance()` directly inside React components. Instead:
1. Create services once at the React app root (in the entry point or a top-level provider)
2. Pass service instances through a React Context (`ServiceContext`)
3. Access services via `useContext(ServiceContext)` in components
4. In tests, wrap components with a `ServiceContext.Provider` that provides mocked services

This also happens to fix the existing TODO about removing the singleton pattern — the React migration is the natural moment to do it.

**Warning signs:**
Tests pass individually but fail when run together (shared singleton state). Changing a mock in `beforeEach` doesn't affect component behavior because the component imported the singleton before the mock was set up. Hot reload during development doesn't pick up service configuration changes.

**Phase to address:**
Phase 1 (scaffold) — establish the service context pattern before any component imports services.

---

### Pitfall 7: Vitest happy-dom Does Not Support WebRTC APIs

**What goes wrong:**
The existing test environment is `happy-dom` (configured in `vitest.config.ts`). `happy-dom` does not implement `RTCPeerConnection`, `navigator.mediaDevices`, `MediaStream`, or `MediaStreamTrack`. Tests for components that use WebRTC APIs throw `TypeError: RTCPeerConnection is not a constructor` or `TypeError: Cannot read properties of undefined (reading 'getUserMedia')` — not a graceful "feature not supported" but a hard crash that fails the entire test suite.

**Why it happens:**
WebRTC APIs are complex browser specifications that test environments (jsdom and happy-dom) have not fully implemented. The existing Lit tests work around this by testing services in isolation with mocked transport layers. React Testing Library tests that render components directly will trigger component initialization code that calls WebRTC APIs.

**How to avoid:**
Add a `vitest.setup.ts` (or extend the existing `test-setup.ts`) with explicit WebRTC API stubs before React tests run:
```typescript
// In test-setup.ts
global.RTCPeerConnection = vi.fn().mockImplementation(() => ({
  addTrack: vi.fn(),
  close: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
}));
Object.defineProperty(navigator, 'mediaDevices', {
  value: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [] }) },
  writable: true,
});
```
Alternatively, use the `@eatsjobs/media-mock` library for more complete MediaStream simulation. For components heavily dependent on WebRTC, prefer testing the custom hook logic in isolation (without rendering) using `renderHook` from `@testing-library/react`.

**Warning signs:**
`TypeError: RTCPeerConnection is not a constructor` in test output. Tests that pass in isolation fail when a WebRTC component is imported in the same test file. Coverage drops to 0% for entire WebRTC-touching modules.

**Phase to address:**
Phase 1 (scaffold) — extend test setup before writing the first React component test.

---

### Pitfall 8: Existing Lit Tests Broken by Adding React/JSX Transform

**What goes wrong:**
Adding `@vitejs/plugin-react` to `vite.config.ts` enables the JSX transform globally. If the Lit components' TypeScript files use decorators (e.g., `@customElement`, `@property`) and the React plugin's Babel transform runs over them, it can conflict with the TypeScript decorator transform. Additionally, `vitest.config.ts` does not include `@vitejs/plugin-react`, meaning the test environment lacks the JSX transform — any test file importing a `.tsx` component crashes with a syntax error.

**Why it happens:**
Vite and Vitest have separate plugin configurations. `vite.config.ts` plugins do not automatically apply to Vitest runs unless `vitest.config.ts` also includes them (or merges from `vite.config.ts`). The React plugin must be added to both configs.

**How to avoid:**
When adding React support:
1. Add `@vitejs/plugin-react` to `vite.config.ts` with file inclusion limited to React-specific files if Lit decorator transforms conflict
2. Add the same plugin to `vitest.config.ts`
3. Run the full Lit test suite immediately after config change — before writing any React code — to confirm no regressions
4. Keep React files in a separate directory (`react/` or `src/react/`) to make glob-based plugin configuration explicit

**Warning signs:**
Existing Lit tests fail after vite.config.ts change with decorator-related errors. New `.tsx` test files throw `Unexpected token` for JSX. Coverage numbers drop without any code changes.

**Phase to address:**
Phase 1 (scaffold) — validated by running existing test suite after scaffold.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Copy DI singleton into React components directly | Faster initial setup | Untestable components, cannot isolate in tests | Never — the TODO exists for a reason |
| Single large AppContext for all state | Avoids designing state structure | Render storms during video, unmaintainable | Never for streaming state; acceptable for settings-only |
| `any` type for gRPC response objects | Skip protobuf type wiring | Loses type safety that is the whole point of protobuf | Only for 1-day spikes, never committed |
| Skip WebRTC cleanup in hooks | Faster to write | Camera leaks, connection accumulation, test failures | Never |
| Test with `// @ts-ignore` around WebRTC mocks | Gets tests passing fast | Masks real API incompatibilities, hides breakage | Never — fix the setup file instead |
| Import Lit component inside React JSX | Reuses existing component | Shadow DOM event bugs, type errors, double rendering | Never — maintain separation by route |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| ConnectRPC client in React | Create transport/client inside component body (recreated every render) | Create once outside component, pass via Context or module-level singleton accessed through Context |
| Vite gRPC proxy | Remove or simplify proxy config when adding React entry | Preserve existing `/grpc` proxy verbatim — React app uses same backend |
| WebSocket alongside ConnectRPC | Assume WebSocket is deprecated, skip it | WebSocket is still live transport for video; React frontend must support it until `StreamProcessVideo` gRPC is implemented |
| Protobuf generated types | Regenerate or copy types for React | Reuse `src/gen/` directly — types are framework-agnostic TypeScript |
| CSS from Lit components | Import Lit CSS files into React | Lit scoped styles use `:host` selectors that don't apply outside shadow DOM — extract shared styles to separate global CSS files |
| OpenTelemetry tracing | Initialize new OTEL instance for React app | Reuse existing telemetry service — reinitializing creates duplicate exporters |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Storing MediaStream in useState | Re-render on every stream update, stale stream references | Store in useRef; only store derived metadata (isActive, trackCount) in state | Immediately — any video component |
| Context value object created inline | All consumers re-render on every parent render | Memoize context value with useMemo; split fast/slow contexts | When parent re-renders frequently (e.g., frame counter) |
| useCallback with wrong dependencies in WebRTC event handlers | Stale closures, old stream referenced | Use useRef for values referenced in callbacks that shouldn't trigger re-creation | When stream changes after initial setup |
| Processing frame data in React state | React batches/defers state updates — frames queue up, display lags | Use requestAnimationFrame with useRef for frame pipeline; bypass React state entirely | At ~15fps and above |
| React DevTools enabled in production | Significant overhead in Observer patterns | Ensure `NODE_ENV=production` build; DevTools is auto-disabled | Development only, but catches people who deploy dev builds |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Exposing gRPC transport object in Context without restriction | Any component can make arbitrary gRPC calls, bypasses use-case layer | Expose only use-case functions through Context, not the raw transport or client |
| Passing camera stream through URL state or localStorage | Stream handle leaks across sessions | MediaStream objects cannot and should not be serialized — keep only in memory via useRef |
| Disabling HTTPS for React dev server | WebRTC requires secure context (HTTPS/localhost) for getUserMedia | Use the existing TLS cert config from vite.config.ts — do not add a plain HTTP fallback |

---

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| No loading state while WebRTC negotiates | Black video box with no feedback for 1-3 seconds | Show explicit "Connecting..." state from first render through ICE completion |
| Error boundaries not wrapping video components | WebRTC error crashes entire React app | Wrap video/streaming components in ErrorBoundary — WebRTC failures are expected (permissions denied, hardware unavailable) |
| React Suspense around WebRTC initialization | "Loading..." flashes cause camera to re-initialize on resume | WebRTC setup is not Suspense-compatible — use explicit state machine (idle/connecting/active/error) |
| Lit CSS bleeding into React route | `:host` styles don't apply but global styles might conflict | Audit global stylesheet for Lit-specific selectors before reusing in React |

---

## "Looks Done But Isn't" Checklist

- [ ] **Video display**: React app shows video — verify camera LED turns off on navigate away (cleanup fires)
- [ ] **gRPC calls**: Filter applies in React UI — verify request goes through the Vite proxy (check Network tab for `/grpc` prefix)
- [ ] **WebSocket transport**: Image processing works — verify WebSocket transport is wired (not just gRPC) since `StreamProcessVideo` is unimplemented
- [ ] **Test suite**: React tests pass — run `npm run test:coverage` and confirm 80% threshold still met for Lit code (existing tests not broken)
- [ ] **Lit frontend**: React route works — navigate to `/lit` and confirm existing frontend still functions (no regressions from config changes)
- [ ] **HMR**: Dev server starts — edit a React component and confirm hot reload fires without full page refresh
- [ ] **Build output**: Dev works — run `npm run build` and confirm production build generates both entry points without hash collisions
- [ ] **Type safety**: No `any` in hook files — run `tsc --noEmit --strict` and confirm zero errors

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Shadow DOM event bugs from mixed Lit/React tree | HIGH | Separate routing fully — remove all Lit elements from React DOM tree; rebuild affected components as pure React |
| Singleton DI imported throughout React components | MEDIUM | Add ServiceContext wrapper at app root; grep for `DIContainer.getInstance()` in React files; replace with `useContext(ServiceContext)` |
| Performance-killing single Context | MEDIUM | Split context by domain; wrap high-frequency consumers with `React.memo`; move stream state to useRef |
| WebRTC cleanup not implemented | LOW | Add cleanup returns to all WebRTC useEffects; test by navigating away with DevTools Memory panel open |
| Vite config broke existing tests | LOW | Revert `vite.config.ts` change; add plugin incrementally; run `npm run test` after each change |
| srcObject assigned as JSX prop | LOW | Find all `<video srcObject=` in JSX; replace with useRef pattern; takes ~15 minutes per component |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|-----------------|--------------|
| srcObject JSX prop failure | Phase: Video streaming component | `chrome://webrtc-internals` shows active connection; video displays live |
| Shadow DOM event bubbling | Phase 1 scaffold — routing architecture | Navigate between `/lit` and `/react`; Lit events don't fire in React console |
| Vite MPA entry point 404 | Phase 1 scaffold — Vite config | `npm run dev` then navigate to `/react` returns 200, not 404 |
| React Context render storms | Phase: useVideoStream hook | React DevTools Profiler shows <5 re-renders/second during active streaming for non-video components |
| WebRTC resource leaks | Phase: useVideoStream hook | Navigate to video page, then away; camera LED off; `chrome://webrtc-internals` shows no active connections |
| DIContainer singleton incompatibility | Phase 1 scaffold — service context | Tests for components pass independently AND in suite (no order dependency) |
| happy-dom WebRTC API missing | Phase 1 scaffold — test setup | `npm run test` passes with zero `TypeError: RTCPeerConnection is not a constructor` |
| React plugin breaks Lit tests | Phase 1 scaffold — Vite config | Full test suite passes after config change (run before first component) |

---

## Sources

- React / srcObject incompatibility: https://github.com/facebook/react/issues/11163 (open since 2017, still unresolved in React 19)
- React shadow DOM event duplication bug: https://github.com/facebook/react/issues/24136
- Lit React integration guide: https://lit.dev/docs/frameworks/react/
- Lit vs React comparison (paradigm differences): https://blog.logrocket.com/lit-vs-react-comparison-guide/
- Vite MPA routing 404 issue: https://github.com/vitejs/vite/discussions/8963
- Vite multiple entry points: https://www.mikemackintosh.com/building-multiple-entrypoints-in-react-using-vite/
- React Context performance pitfalls: https://blog.logrocket.com/pitfalls-of-overusing-react-context/
- WebRTC cleanup in React useEffect: https://www.videosdk.live/developer-hub/webrtc/react-webrtc-video-call
- React Testing Library flaky tests: https://kentcdodds.com/blog/common-mistakes-with-react-testing-library
- WebRTC mock library for Vitest/Jest: https://jsr.io/@eatsjobs/media-mock
- Shadow DOM event retargeting: https://pm.dartus.fr/posts/2021/shadow-dom-and-event-propagation/
- ConnectRPC web client setup: https://connectrpc.com/docs/web/using-clients/
- Codebase-specific: `.planning/codebase/CONCERNS.md` — DIContainer singleton TODO, WebSocket/gRPC dual protocol, video processing latency

---
*Pitfalls research for: Lit-to-React migration, WebRTC in React, Vite MPA, ConnectRPC React integration*
*Researched: 2026-04-12*
