# Phase 03 Research: Static Feature UI

**Phase:** 03-static-feature-ui
**Research Date:** 2026-04-13
**Discovery Level:** Level 2 (Standard Research - Choosing between existing Lit patterns and new React implementations)

## Overview

Phase 3 implements all static UI features for the React frontend, matching functionality from the existing Lit frontend:
- Image processing workflow (upload, filter selection, result display)
- File management (list and select previously uploaded images)
- System settings (view and modify configuration)
- Health monitoring (status display and visual indicators)

This research documents the existing Lit implementations, gRPC APIs, and React patterns to guide the React component development.

## Existing Lit Implementations

### 1. Image Upload (`image-upload.ts`)

**Location:** `front-end/src/components/image/image-upload.ts`

**Key Features:**
- Drag-and-drop and click-to-upload
- File validation: PNG only, max 10MB
- Progress bar during upload (simulated with interval + actual progress)
- Dispatches `image-uploaded` event with `StaticImage` result
- Uses `fileService.uploadImage()` from `file-service.ts`

**Service Layer:**
```typescript
// FileService interface
interface IFileService {
  listAvailableImages(): Promise<StaticImage[]>;
  uploadImage(file: File): Promise<StaticImage>;
}
```

**gRPC API:** `FileService.uploadImage`
- Request: `{ filename: string, fileData: Uint8Array }`
- Response: `{ image: StaticImage, message: string }`

**StaticImage Type (from `config_service_pb`):**
```typescript
class StaticImage {
  id: string;
  displayName: string;
  path: string;  // URL to image
  isDefault: boolean;
}
```

### 2. Filter Panel (`filter-panel.ts`)

**Location:** `front-end/src/components/app/filter-panel.ts`

**Key Features:**
- Displays list of available filters
- Toggle enable/disable per filter
- Drag-and-drop reordering
- Expandable filter cards with parameter controls
- Parameter types: select, slider/range, number, checkbox
- Dispatches `filter-change` event with active filters
- Validation with toast notifications for number inputs

**Filter Data Structure:**
```typescript
interface Filter {
  id: string;
  name: string;
  enabled: boolean;
  expanded: boolean;
  parameters: GenericFilterParameter[];
  parameterValues: { [key: string]: string };
}

interface GenericFilterParameter {
  id: string;
  name: string;
  type: GenericFilterParameterType; // SELECT, RANGE, NUMBER, CHECKBOX, TEXT
  options: GenericFilterParameterOption[];
  defaultValue: string;
  metadata: { [key: string]: string };
}
```

**gRPC API:** `ImageProcessorService.listFilters`
- Request: `ListFiltersRequest {}`
- Response: `ListFiltersResponse { filters: GenericFilterDefinition[] }`

**GenericFilterDefinition:** Contains id, name, type, and parameters

### 3. Image Selector Modal (`image-selector-modal.ts`)

**Location:** `front-end/src/components/image/image-selector-modal.ts`

**Key Features:**
- Grid display of available images
- Image preview thumbnails
- Click to select
- Dispatches `image-selected` event
- Uses `fileService.listAvailableImages()`

### 4. Source Drawer (`source-drawer.ts`)

**Location:** `front-end/src/components/app/source-drawer.ts`

**Key Features:**
- Slide-out drawer for input source selection
- Tabs for images vs videos
- Integrates image upload and image selector
- Dispatches `source-selected` event with `InputSource`

### 5. Connection Status (`connection-status-card.ts`)

**Location:** `front-end/src/components/app/connection-status-card.ts`

**Key Features:**
- Visual indicator (colored dot: green/red/orange)
- Status display with tooltip
- Last request information
- Connection details (protocol, endpoint)

### 6. gRPC Unavailable Page (`grpc-unavailable.ts`)

**Location:** `front-end/src/components/app/grpc-unavailable.ts`

**Key Features:**
- Informational content when backend is down
- Start Jetson Nano button
- Terminal-style progress display
- Uses `remoteManagementService.startJetsonNano()`

### 7. Feature Flags Modal (`feature-flags-modal.ts`)

**Location:** `front-end/src/components/flags/feature-flags-modal.ts`

**Key Features:**
- Modal pattern with backdrop and fade/scale animations
- Iframe integration for Flipt UI
- Sync button calling `ConfigService.syncFeatureFlags`
- Toast notifications for success/error

## React Patterns Established in Phase 2

### 1. Service Context Pattern

```typescript
// context/service-context.tsx
interface GrpcClients {
  imageProcessorClient: PromiseClient<typeof ImageProcessorService>;
  remoteManagementClient: PromiseClient<typeof RemoteManagementService>;
}

const ServiceContext = createContext<GrpcClients | null>(null);

export function useServiceContext(): GrpcClients {
  const context = useContext(ServiceContext);
  if (!context) throw new Error('useServiceContext must be used within GrpcClientsProvider');
  return context;
}
```

### 2. Custom Hook Pattern

```typescript
// hooks/useFilters.ts
export function useFilters() {
  const { imageProcessorClient } = useServiceContext();
  const [filters, setFilters] = useState<GenericFilterDefinition[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<GrpcAsyncError | null>(null);

  const refetch = useCallback(() => {
    // Async logic with AbortController
  }, []);

  useEffect(() => {
    refetch();
    return () => {
      abortRef.current?.abort();
    };
  }, [refetch]);

  return { filters, loading, error, refetch };
}
```

**Key Patterns:**
- `useRef` for clients to avoid stale closures
- `AbortController` for cancellation
- Request generation guard to handle race conditions
- Cleanup in `useEffect` return
- Returns `{ data, loading, error, refetch }`

### 3. Toast Pattern

```typescript
// hooks/useToast.ts
export function useToast() {
  const context = useContext(ToastContext);
  if (!context) {
    console.warn('useToast called without ToastProvider');
    return { success: () => {}, error: () => {} };
  }
  return context;
}
```

**Usage:**
```typescript
const { success, error } = useToast();
success('Title', 'Message');
error('Title', 'Message');
```

## gRPC API Reference

### ImageProcessorService

**1. listFilters**
```typescript
interface ListFiltersRequest {
  traceContext?: TraceContext;
}

interface ListFiltersResponse {
  filters: GenericFilterDefinition[];
  traceContext?: TraceContext;
}
```

**2. processImage**
```typescript
interface ProcessImageRequest {
  imagePath: string;
  filters: ActiveFilter[];
  traceContext?: TraceContext;
}

interface ActiveFilter {
  id: string;
  parameters: { [key: string]: string };
}

interface ProcessImageResponse {
  success: boolean;
  processedImagePath: string;
  message: string;
  traceContext?: TraceContext;
}
```

### FileService

**1. listAvailableImages**
```typescript
interface ListAvailableImagesRequest {
  traceContext?: TraceContext;
}

interface ListAvailableImagesResponse {
  images: StaticImage[];
  traceContext?: TraceContext;
}
```

**2. uploadImage**
```typescript
interface UploadImageRequest {
  filename: string;
  fileData: Uint8Array;
}

interface UploadImageResponse {
  image?: StaticImage;
  message: string;
}
```

### ConfigService

**1. getStreamConfig**
```typescript
interface GetStreamConfigRequest {
  traceContext?: TraceContext;
}

interface GetStreamConfigResponse {
  endpoints: StreamEndpoint[];
  traceContext?: TraceContext;
}

interface StreamEndpoint {
  type: string;
  endpoint: string;
  transportFormat: string;
  logLevel: string;
  consoleLogging: boolean;
}
```

**2. getSystemInfo** (for settings display)
```typescript
interface GetSystemInfoRequest {
  traceContext?: TraceContext;
}

interface GetSystemInfoResponse {
  systemInfo: SystemInfo;
  traceContext?: TraceContext;
}
```

### RemoteManagementService

**1. checkAcceleratorHealth** (already in useHealthMonitor)
```typescript
interface CheckAcceleratorHealthRequest {
  traceContext?: TraceContext;
}

interface CheckAcceleratorHealthResponse {
  status: AcceleratorHealthStatus; // HEALTHY, UNHEALTHY, UNKNOWN
  message: string;
  traceContext?: TraceContext;
}
```

## React Component Architecture

### Component Structure

```
front-end/src/react/
├── components/
│   ├── image/
│   │   ├── ImageUpload.tsx
│   │   ├── ImageSelector.tsx
│   │   └── ImageProcessor.tsx
│   ├── filters/
│   │   └── FilterPanel.tsx
│   ├── files/
│   │   └── FileList.tsx
│   ├── settings/
│   │   └── SettingsPanel.tsx
│   └── health/
│       └── HealthIndicator.tsx
├── hooks/
│   ├── useImageUpload.ts
│   ├── useImageProcessing.ts
│   └── useConfig.ts
└── App.tsx (updated with feature routing)
```

### State Management Strategy

**No external state management library required (per project scope).**
- Use React Context for global state (already have ServiceContext, ToastContext)
- Use component state for local UI state
- Use custom hooks for data fetching and complex logic

### Styling Approach

**Decision needed:** CSS framework or CSS modules?
- Lit uses CSS custom properties and inline styles in `static styles`
- React options:
  1. **CSS Modules** (recommended for React)
  2. **Tailwind CSS** (not in package.json yet)
  3. **Styled Components** (not in package.json)
  4. **Plain CSS with CSS custom properties** (matches Lit pattern)

**Recommendation:** Use CSS Modules for React components to maintain parity with Lit styling while following React best practices. Reuse CSS custom properties for theme consistency.

## Task Breakdown Strategy

### Plan 01: Image Upload & Filter Selection (IMG-01, IMG-02)
- `ImageUpload.tsx` component
- `FilterPanel.tsx` component
- `useImageUpload.ts` hook
- `useFilters` hook (already exists, may need enhancement)

### Plan 02: Image Processing & Results (IMG-03, IMG-04)
- `ImageProcessor.tsx` component
- `useImageProcessing.ts` hook
- Progress tracking for upload and processing

### Plan 03: File Management (FILE-01, FILE-02)
- `FileList.tsx` component
- `ImageSelector.tsx` component (modal)
- `useFiles.ts` hook

### Plan 04: Settings & Configuration (CONF-01, CONF-02)
- `SettingsPanel.tsx` component
- `useConfig.ts` hook
- Form handling for configuration updates

### Plan 05: Health Monitoring UI (HLTH-01, HLTH-02)
- `HealthIndicator.tsx` component
- Integration with existing `useHealthMonitor` hook
- Visual status indicators

## Dependency Graph

```
Plan 01 (Image Upload & Filters)
  → Creates: ImageUpload, FilterPanel, useImageUpload
  → Depends on: Phase 2 (useFilters, useToast)

Plan 02 (Image Processing)
  → Creates: ImageProcessor, useImageProcessing
  → Depends on: Plan 01 (filters from FilterPanel)

Plan 03 (File Management)
  → Creates: FileList, ImageSelector, useFiles
  → Depends on: Phase 2 (useAsyncGRPC)

Plan 04 (Settings)
  → Creates: SettingsPanel, useConfig
  → Depends on: Phase 2 (useAsyncGRPC, useToast)

Plan 05 (Health UI)
  → Creates: HealthIndicator
  → Depends on: Phase 2 (useHealthMonitor)
```

**Parallel Execution:**
- Wave 1: Plan 01, Plan 03, Plan 04, Plan 05 (all independent)
- Wave 2: Plan 02 (depends on Plan 01)

## Testing Strategy

### Unit Tests (Vitest + React Testing Library)
- Component rendering with correct props
- User interactions (click, drag, input)
- Hook behavior with mocked gRPC clients
- Error states and loading states

### Integration Points
- gRPC client mocking with `renderWithService`
- Toast notifications verification
- Event dispatching and handling

## Open Questions

1. **Styling approach:** CSS Modules vs Tailwind vs plain CSS?
   - **Recommendation:** CSS Modules to match React patterns, reuse Lit CSS custom properties

2. **Modal library:** Use existing pattern or install library?
   - **Recommendation:** Implement custom modal component following `feature-flags-modal.ts` pattern (no new dependencies)

3. **Form validation:** Use library or custom?
   - **Recommendation:** Custom validation with React state (simple use case, no new dependencies)

4. **Image processing result display:** URL from gRPC or base64?
   - **Research finding:** gRPC returns `processedImagePath` (URL string)

## Security Considerations

### Trust Boundaries
- **Client → gRPC API:** All user inputs (files, filter parameters)
- **File upload:** File size and type validation on client (defense in depth)
- **Filter parameters:** Parameter validation on client + server-side validation

### STRIDE Threats

| Threat | Component | Mitigation |
|--------|-----------|------------|
| Tampering | File upload | Validate file type (PNG only), max size (10MB) |
| Tampering | Filter parameters | Validate number ranges, enum values |
| Information Disclosure | Error messages | Sanitize error messages before display |
| Denial of Service | Large file upload | 10MB limit on client + server |

## Conclusion

Phase 3 has clear Lit reference implementations and established React patterns from Phase 2. All required gRPC APIs are already defined and in use by the Lit frontend. No new dependencies are required if using CSS Modules for styling.

**Estimated complexity:** Medium
**Estimated number of plans:** 5
**Estimated execution time:** 2-3 hours
