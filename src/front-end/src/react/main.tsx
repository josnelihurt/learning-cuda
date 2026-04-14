import './react-root.css';
import '@/components/app/information-banner';
import '@/components/flags/feature-flags-modal';
import '@/components/app/grpc-status-modal';
import '@/components/app/app-tour';
import '@/components/ui/tools-dropdown';
import '@/components/flags/feature-flags-button';
import '@/components/flags/sync-flags-button';
import '@/components/ui/version-tooltip-lit';
import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { ToastProvider } from './context/toast-context';
import { DashboardStateProvider } from './context/dashboard-state-context';
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
        <DashboardStateProvider>
          <GrpcClientsProvider>
            <App />
          </GrpcClientsProvider>
        </DashboardStateProvider>
      </AppServicesProvider>
    </ToastProvider>
  </StrictMode>
);
