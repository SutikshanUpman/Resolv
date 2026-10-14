import { ButtonLink } from '../components/ui/ButtonLink';
import { PageMeta } from '../components/layout/PageMeta';

export function NotFoundPage() {
  return (
    <>
      <PageMeta title="Page not found" />
      <section className="container-page flex flex-col items-start pt-28 pb-16 md:pt-36">
        <p className="font-mono text-sm text-glow">404</p>
        <h1 className="mt-4 text-h1 font-semibold">No record matches this page.</h1>
        <p className="mt-5 max-w-xl text-lead text-muted">The address may be mistyped, or the page has moved.</p>
        <div className="mt-9 flex flex-wrap gap-3">
          <ButtonLink to="/">Back to the start</ButtonLink>
          <ButtonLink to="/architecture/docs" variant="secondary">
            Read the docs
          </ButtonLink>
        </div>
      </section>
    </>
  );
}
