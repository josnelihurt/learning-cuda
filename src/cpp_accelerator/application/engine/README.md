# Engine Filter Creation Dispatch

## Pattern

This module uses **Strategy Pattern with callable dispatch** for `IFilterFactory::CreateFilter`.

- The shared dispatch flow lives in `filter_creation_dispatch.hpp`.
- Each backend factory injects backend-specific strategies as lambdas.

Conceptually, this keeps a single `FilterType` switch (Template Method-like flow) while preserving per-backend construction behavior.

## Why this exists

- Avoids duplicating the same `switch (FilterType)` in every factory.
- Keeps backend-specific parameter adaptation where it belongs (inside each factory lambda).
- Keeps the API simple: no `std::function` indirection and no generic utility bucket.

## How to use

Inside a factory `CreateFilter(type, params)`:

1. Call `DispatchCreateFilter(type, on_grayscale, on_blur)`.
2. Implement each callable with backend-specific construction logic.
3. Keep any enum/parameter mapping local to the relevant callable.

Example shape:

```cpp
return DispatchCreateFilter(
    type,
    [&params]() { return std::make_unique<MyGrayFilter>(params.grayscale_algorithm); },
    [&params]() {
      const auto backend_mode = MapMode(params.blur_border_mode);
      return std::make_unique<MyBlurFilter>(params.blur_kernel_size, params.blur_sigma, backend_mode,
                                            params.blur_separable);
    });
```

## Do and don't

- Do keep dispatch limited to common `FilterType` routing.
- Do keep backend-specific logic in factory lambdas.
- Do return `nullptr` for unsupported filter types via dispatcher default path.
- Don't move backend mapping logic into the shared dispatcher.
- Don't turn this into a registry of dynamic callbacks unless a real requirement appears.
