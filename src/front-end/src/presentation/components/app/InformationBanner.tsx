import { useState, type ReactElement } from 'react';
import styles from './InformationBanner.module.css';

export function InformationBanner(): ReactElement {
  const [isVisible, setIsVisible] = useState(true);
  return (
    <>
      {isVisible && (
        <div className={styles.banner} onClick={() => setIsVisible((value) => !value)}>
          <div className={styles.text}>
            Production deployment in progress - some components may be unavailable click to close
          </div>
        </div>
      )}
    </>
  );
}
