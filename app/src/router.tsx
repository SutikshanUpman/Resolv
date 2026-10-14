import { createBrowserRouter } from 'react-router';
import { AppShell } from './components/layout/AppShell';
import { ShellFallback } from './components/layout/ShellFallback';
import { RootErrorPage } from './pages/RootErrorPage';
import { RouteErrorPage } from './pages/RouteErrorPage';

// Every page is its own lazily loaded chunk.
export const router = createBrowserRouter([
  {
    Component: AppShell,
    ErrorBoundary: RootErrorPage,
    HydrateFallback: ShellFallback,
    children: [
      {
        // Page errors render inside the shell, so navigation stays available.
        ErrorBoundary: RouteErrorPage,
        children: [
          { index: true, lazy: async () => ({ Component: (await import('./pages/LandingPage')).LandingPage }) },
          { path: 'architecture', lazy: async () => ({ Component: (await import('./pages/ArchitecturePage')).ArchitecturePage }) },
          { path: 'architecture/docs', lazy: async () => ({ Component: (await import('./pages/DocsPage')).DocsPage }) },
          { path: 'app', lazy: async () => ({ Component: (await import('./pages/UploadPage')).UploadPage }) },
          { path: 'app/run', lazy: async () => ({ Component: (await import('./pages/RunPage')).RunPage }) },
          { path: 'app/results', lazy: async () => ({ Component: (await import('./pages/ResultsPage')).ResultsPage }) },
          { path: '*', lazy: async () => ({ Component: (await import('./pages/NotFoundPage')).NotFoundPage }) },
        ],
      },
    ],
  },
]);
