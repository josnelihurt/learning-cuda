import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { GrpcClientsProvider } from './providers/grpc-clients-provider';

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error('Root element not found');
}

createRoot(rootElement).render(
  <StrictMode>
    <GrpcClientsProvider>
      <App />
    </GrpcClientsProvider>
  </StrictMode>
);
