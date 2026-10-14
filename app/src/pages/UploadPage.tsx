import { useState } from 'react';
import { useNavigate } from 'react-router';
import { PageHeader } from '../components/layout/PageHeader';
import { PageMeta } from '../components/layout/PageMeta';
import { SourceDropZone } from '../components/workspace/SourceDropZone';
import { Button } from '../components/ui/Button';
import { resolver } from '../lib/resolver';
import { SOURCES, loadSample } from '../lib/workspace/loadSources';
import { hasUpload, isReady, useWorkspace } from '../lib/workspace/store';

export function UploadPage() {
  const ws = useWorkspace();
  const navigate = useNavigate();
  const [notice, setNotice] = useState(false);
  const ready = isReady(ws);
  const uploaded = hasUpload(ws);
  const loading = Object.values(ws.slots).some((s) => s.status === 'parsing');

  const refine = () => {
    // The demo never runs models: a visitor's own files stop here, after real parsing and checks.
    if (uploaded && !resolver.liveInference) {
      setNotice(true);
      return;
    }
    void navigate('/app/run');
  };

  return (
    <>
      <PageMeta title="Workspace" />
      <PageHeader
        eyebrow="Workspace"
        title="Drop in three sources"
        lead="A reference list of businesses, and two sources with records to match against it. Or start with real data from our submission."
      />

      <section className="container-page">
        <div className="flex flex-col items-start justify-between gap-5 rounded-card border border-accent/40 bg-accent/10 p-6 md:flex-row md:items-center">
          <div>
            <h2 className="font-semibold text-ink">Try with sample data</h2>
            <p className="mt-1 max-w-2xl text-sm text-muted">
              Real businesses from the challenge test set, sampled evenly across France, India and the US, with every candidate record our
              pipeline considered. Results are replayed from our real submission.
            </p>
          </div>
          <Button size="lg" onClick={() => void loadSample()} disabled={loading}>
            {loading ? 'Loading…' : 'Load sample data'}
          </Button>
        </div>

        <div className="mt-6 grid gap-4 lg:grid-cols-3">
          {SOURCES.map((s) => (
            <SourceDropZone key={s.id} source={s.id} label={s.label} role={s.role} slot={ws.slots[s.id]} />
          ))}
        </div>

        {notice && uploaded && (
          <div role="alert" className="mt-6 rounded-card border border-unsure/40 bg-unsure/10 p-6">
            <h2 className="font-semibold text-ink">Live inference isn’t connected</h2>
            <p className="mt-2 max-w-3xl text-sm text-ink/85">
              Your files were read and checked in your browser, and nothing was uploaded anywhere. The trained models aren’t part of
              this demo, so they can’t run on your data. You can explore the full flow with the sample data instead.
            </p>
            <Button className="mt-4" onClick={() => void loadSample().then(() => setNotice(false))}>
              Use the sample data
            </Button>
          </div>
        )}

        <div className="mt-8 flex flex-col items-start gap-3 border-t border-line pt-8 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-sm text-muted">
            {ready
              ? uploaded
                ? 'All three files passed the checks.'
                : 'Sample data is ready.'
              : 'Add all three sources to continue.'}
          </p>
          <Button size="lg" disabled={!ready} onClick={refine}>
            Refine
          </Button>
        </div>
      </section>
    </>
  );
}
