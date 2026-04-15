# Step 7 - Frontend transport via WebRTC

## Status
done

## Outcome
- Replaced the active frontend frame path with `WebRTCFrameTransportService`.
- Switched the active React and Lit video grids away from direct WebSocket frame transport usage.
- Removed the dead WebSocket-era frame transport/service files from the frontend.
- Updated shared transport interfaces and connection/status UI models to reflect WebRTC/data-channel transport.

## Validation
- `cd src/front-end && npm run build`
- `cd src/front-end && npm run test -- --run`
- `bazel build //src/cpp_accelerator/ports/grpc:image_processor_grpc_server`
- `cd src/go_api && make build`
- `./scripts/test/unit-tests.sh`

## Notes
- Frontend frame payloads now use `ProcessImageRequest` / `ProcessImageResponse` over the browser WebRTC data channel.
- Unary playback control uses `StartVideoPlayback` / `StopVideoPlayback`.
- This step is complete for the active frontend transport path, but Step 8 still tracks repo-wide cleanup and the remaining signaling limitation.
