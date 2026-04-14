import { useEffect } from 'react';
import { VideoStreamer } from './components/video/VideoStreamer';
import { useAppServices } from './providers/app-services-provider';

function CreditByClickHandler() {
  useEffect(() => {
    const el = document.getElementById('credit-by');
    if (!el) {
      return;
    }
    const handler = () => {
      localStorage.clear();
      console.log('LocalStorage cleared');
      alert('LocalStorage cleared!');
    };
    el.addEventListener('click', handler);
    return () => el.removeEventListener('click', handler);
  }, []);
  return null;
}

function MainContent() {
  const { ready } = useAppServices();

  return (
    <main className="main-content" data-testid="react-app-ready">
      {ready ? (
        <VideoStreamer />
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
            <span className="navbar-badge">React app loaded</span>
          </div>
          <div className="navbar-credit">
            <span id="credit-by" style={{ cursor: 'pointer' }} title="Click to clear localStorage">
              by
            </span>{' '}
            <a href="https://josnelihurt.me" target="_blank" rel="noreferrer">
              josnelihurt
            </a>
          </div>
        </div>
        <information-banner />
      </header>

      <aside className="sidebar">
        <div className="sidebar-content">
          <div data-testid="sidebar-shell-placeholder" />
        </div>
      </aside>

      <MainContent />
      <CreditByClickHandler />
    </>
  );
}

export default App;
