import { useEffect, useState } from 'react';
import { AcceleratorType } from '@/gen/common_pb';
import { controlChannelService } from '@/infrastructure/transport/control-channel-service';
import { logger } from '@/infrastructure/observability/otel-logger';

export type AcceleratorOption = { label: string; value: AcceleratorType };

export function useAcceleratorCapabilities() {
  const [options, setOptions] = useState<AcceleratorOption[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    controlChannelService
      .getAcceleratorCapabilities()
      .then((response) => {
        if (cancelled) {
          return;
        }
        const nextOptions: AcceleratorOption[] = response.supportedOptions.map((option) => ({
          label: `${response.displayName}: ${option.label}@v${response.version}`,
          value: option.type,
        }));
        setOptions(nextOptions);
      })
      .catch((error) => {
        if (!cancelled) {
          logger.warn('Failed to load accelerator capabilities', {
            'error.message': error instanceof Error ? error.message : String(error),
          });
          setOptions([]);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { options, loading };
}
