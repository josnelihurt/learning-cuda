import { useState } from 'react';
import './InformationBanner.css';

export function InformationBanner() {
  const [isVisible, setIsVisible] = useState(true);
  return (
    <>
      {isVisible && (
        <div className="information-banner" onClick={() => setIsVisible((value) => !value)}>
          <div className="information-banner__text">
            Production deployment in progress - some components may be unavailable click to close
          </div>
        </div>
      )}
    </>
  );
}
