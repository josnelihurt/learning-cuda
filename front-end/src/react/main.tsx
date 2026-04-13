import '@/components/app/toast-container';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { ToastProvider } from './context/toast-context';
import { GrpcClientsProvider } from './providers/grpc-clients-provider';

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error('Root element not found');
}

createRoot(rootElement).render(
  <StrictMode>
    <ToastProvider>
      <GrpcClientsProvider>
        <App />
      </GrpcClientsProvider>
    </ToastProvider>
  </StrictMode>
);
