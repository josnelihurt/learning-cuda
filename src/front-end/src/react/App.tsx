import React, { useEffect, useState } from 'react';
import { VideoGridHost } from './components/video/VideoGridHost';
import { SidebarColumn } from './components/sidebar/SidebarColumn';
import { useAppServices } from './providers/app-services-provider';
import { ReactNavbarControls } from './components/app/ReactNavbarControls';
import { ReactFeatureFlagsModal } from './components/app/ReactFeatureFlagsModal';
import { ReactGrpcStatusModal } from './components/app/ReactGrpcStatusModal';
import { ReactAppTour } from './components/app/ReactAppTour';
import { ReactInformationBanner } from './components/app/ReactInformationBanner';

function MainContent() {
  const { ready } = useAppServices();

  return (
    <main className="main-content" data-testid="react-app-ready">
      {ready ? (
        <VideoGridHost />
      ) : (
        <div className="react-main-loading" data-testid="react-main-loading">
          Loading services…
        </div>
      )}
    </main>
  );
}

export function App() {
  const [isFeatureFlagsOpen, setIsFeatureFlagsOpen] = useState(false);

  useEffect(() => {
    const clearStorage = () => {
      localStorage.clear();
      alert('LocalStorage cleared!');
    };
    const creditBy = document.getElementById('credit-by');
    creditBy?.addEventListener('click', clearStorage);
    return () => {
      creditBy?.removeEventListener('click', clearStorage);
    };
  }, []);

  return (
    <>
      <header className="navbar">
        <div className="navbar-container">
          <div className="navbar-left">
            <a href="/" className="navbar-brand">
              <span className="accent">CUDA</span> Image Processor
            </a>
          </div>
          <div className="navbar-services" data-testid="navbar-services-placeholder">
            <ReactNavbarControls onOpenFeatureFlags={() => setIsFeatureFlagsOpen(true)} />
            <span className="navbar-badge">React app loaded</span>
          </div>
          <div className="navbar-credit">
            <span id="credit-by" className="clickable-credit" title="Click to clear localStorage">
              by
            </span>{' '}
            <a href="https://josnelihurt.me" target="_blank" rel="noreferrer">
              josnelihurt
            </a>
          </div>
        </div>
        <ReactInformationBanner />
      </header>

      <SidebarColumn />

      <MainContent />
      <ReactFeatureFlagsModal isOpen={isFeatureFlagsOpen} onClose={() => setIsFeatureFlagsOpen(false)} />
      <ReactGrpcStatusModal />
      <ReactAppTour />
    </>
  );
}

export default App;
