import { VideoStreamer } from './components/video/VideoStreamer';
import { HealthIndicator } from './components/health/HealthIndicator';
import { useHealthMonitor } from './hooks/useHealthMonitor';
import styles from './App.module.css';

export function App() {
  const { isHealthy, loading } = useHealthMonitor();

  return (
    <div className={styles.app}>
      <header className="navbar">
        <div className="navbar-container">
          <div className="navbar-left">
            <a href="/" className="navbar-brand">
              <span className="accent">CUDA</span> Image Processor
            </a>
          </div>
          <div className="navbar-services">
            <HealthIndicator isHealthy={isHealthy} loading={loading} />
            <span className="navbar-badge">React</span>
          </div>
        </div>
      </header>
      <main className="main-content">
        <VideoStreamer />
      </main>
    </div>
  );
}

export default App;
