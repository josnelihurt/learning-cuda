import React, { useEffect, useState } from 'react';
import { VideoGridHost } from './components/video/VideoGridHost';
import { SidebarColumn } from './components/sidebar/SidebarColumn';
import { useAppServices } from './providers/app-services-provider';
import { NavbarControls, VersionTooltip } from './components/app/NavbarControls';
import { FeatureFlagsModal } from './components/app/FeatureFlagsModal';
import { GrpcStatusModal } from './components/app/GrpcStatusModal';
import { AppTour } from './components/app/AppTour';
import { InformationBanner } from './components/app/InformationBanner';

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
            <NavbarControls onOpenFeatureFlags={() => setIsFeatureFlagsOpen(true)} />
          </div>
          <div className="navbar-credit">
            <span id="credit-by" className="clickable-credit" title="Click to clear localStorage">
              by
            </span>{' '}
            <a href="https://josnelihurt.me" target="_blank" rel="noreferrer">
              josnelihurt
            </a>
            <VersionTooltip />
          </div>
        </div>
        <InformationBanner />
      </header>

      <SidebarColumn />

      <MainContent />
      <FeatureFlagsModal isOpen={isFeatureFlagsOpen} onClose={() => setIsFeatureFlagsOpen(false)} />
      <GrpcStatusModal />
      <AppTour />
    </>
  );
}

export default App;
