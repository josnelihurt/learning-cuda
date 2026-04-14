import { VideoGridHost } from './components/video/VideoGridHost';
import { SidebarColumn } from './components/sidebar/SidebarColumn';
import { useAppServices } from './providers/app-services-provider';

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
              <tools-dropdown></tools-dropdown>
              <feature-flags-button></feature-flags-button>
              <sync-flags-button style={{ display: 'none' }}></sync-flags-button>
              <span className="navbar-badge">React app loaded</span>
            </div>
            <div className="navbar-credit">
              <span id="credit-by" style={{ cursor: 'pointer' }} title="Click to clear localStorage">
                by
              </span>{' '}
              <a href="https://josnelihurt.me" target="_blank" rel="noreferrer">
                josnelihurt
              </a>
              <version-tooltip-lit>
                <button type="button" className="info-btn" title="Version Information">
                  <span>i</span>
                </button>
              </version-tooltip-lit>
            </div>
          </div>
          <information-banner />
        </header>

        <SidebarColumn />

        <MainContent />
    </>
  );
}

export default App;
