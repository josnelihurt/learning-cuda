import type { InputSource } from '@/gen/config_service_pb';

export async function isWebcamUsable(): Promise<boolean> {
  if (typeof navigator === 'undefined' || !navigator.mediaDevices?.enumerateDevices) {
    return false;
  }
  try {
    const permissions = navigator.permissions as Permissions | undefined;
    if (permissions?.query) {
      try {
        const status = await permissions.query({ name: 'camera' as PermissionName });
        if (status.state === 'denied') return false;
      } catch {
        // Some browsers (Firefox) don't support 'camera' here — fall through to enumerateDevices.
      }
    }
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.some((d) => d.kind === 'videoinput');
  } catch {
    return false;
  }
}

export async function pickDefaultSource(sources: InputSource[]): Promise<InputSource | undefined> {
  const remoteCamera = sources.find((s) => s.type === 'remote_camera');
  if (remoteCamera) return remoteCamera;

  const webcam = sources.find((s) => s.type === 'camera');
  if (webcam && (await isWebcamUsable())) return webcam;

  return sources.find((s) => s.type === 'static') ?? sources.find((s) => s.isDefault);
}

export function effectiveAutoselectSourceId(
  sources: InputSource[],
  webcamUsable: boolean
): string | undefined {
  const remote = sources.find((s) => s.type === 'remote_camera');
  if (remote) return remote.id;
  const cam = sources.find((s) => s.type === 'camera');
  if (cam && webcamUsable) return cam.id;
  return sources.find((s) => s.type === 'static')?.id;
}

export function gridSourceDisplayName(inputSource: InputSource): string {
  if (inputSource.type === 'remote_camera') {
    const n = inputSource.displayName?.trim();
    return n || 'Remote camera';
  }
  if (inputSource.type === 'camera') {
    const n = inputSource.displayName?.trim();
    if (!n || n.toLowerCase() === 'webcam') return 'Camera';
    return n;
  }
  return inputSource.displayName?.trim() ?? '';
}
