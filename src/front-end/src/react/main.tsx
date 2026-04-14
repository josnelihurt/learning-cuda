import './react-root.css';
import '@/components/app/toast-container';
import '@/components/app/information-banner';
import '@/components/app/stats-panel';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { ToastProvider } from './context/toast-context';
import { AppServicesProvider } from './providers/app-services-provider';
import { GrpcClientsProvider } from './providers/grpc-clients-provider';

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error('Root element not found');
}

createRoot(rootElement).render(
  <StrictMode>
    <ToastProvider>
      <AppServicesProvider>
        <GrpcClientsProvider>
          <App />
        </GrpcClientsProvider>
      </AppServicesProvider>
    </ToastProvider>
  </StrictMode>
);
