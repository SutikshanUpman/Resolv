import { isRouteErrorResponse, useRouteError } from 'react-router';
import { Button } from '../components/ui/Button';
import { ButtonLink } from '../components/ui/ButtonLink';
import { PageMeta } from '../components/layout/PageMeta';

/** A page failed to load or render; shown inside the shell so the navigation still works. */
export function RouteErrorPage() {
  const error = useRouteError();
  const detail = isRouteErrorResponse(error)
    ? `${error.status} ${error.statusText}`
    : error instanceof Error
      ? error.message
      : 'Unknown error';

  return (
    <>
      <PageMeta title="Something went wrong" />
      <section className="container-page pt-28 pb-16 md:pt-36" role="alert">
        <p className="font-mono text-sm text-danger">Error</p>
        <h1 className="mt-4 text-h1 font-semibold">This page didn’t load.</h1>
        <p className="mt-5 max-w-xl text-lead text-muted">
          A newer version of the site may have been published since you opened it. Reloading usually fixes this.
        </p>
        <p className="mt-4 font-mono text-xs text-muted">{detail}</p>
        <div className="mt-9 flex flex-wrap gap-3">
          <Button onClick={() => window.location.reload()}>Reload</Button>
          <ButtonLink to="/" variant="secondary">
            Back to the start
          </ButtonLink>
        </div>
      </section>
    </>
  );
}
