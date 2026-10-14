import { useRouteError } from 'react-router';

/** Last resort when the shell itself fails: no router components, plain markup only. */
export function RootErrorPage() {
  const error = useRouteError();
  return (
    <main className="container-page grid min-h-dvh place-content-center gap-4 text-center" role="alert">
      <h1 className="text-h2 font-semibold">Something went wrong.</h1>
      <p className="text-muted">{error instanceof Error ? error.message : 'The page could not be shown.'}</p>
      <p>
        <a href="/" className="text-glow underline underline-offset-4">
          Reload the site
        </a>
      </p>
    </main>
  );
}
