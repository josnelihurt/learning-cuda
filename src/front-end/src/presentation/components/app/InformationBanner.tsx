import { useEffect, useState, type ReactElement } from 'react';
import styles from './InformationBanner.module.css';
import { useDashboardState } from '@/presentation/context/dashboard-state-context';

export function InformationBanner(): ReactElement {
  const { isWebRTCReady } = useDashboardState();
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    if (isWebRTCReady) setIsVisible(false);
  }, [isWebRTCReady]);

  if (!isVisible) return <></>;

  return (
    <div className={styles.banner} onClick={() => setIsVisible(false)}>
      <div className={styles.text}>Connecting... please wait</div>
    </div>
  );
}
