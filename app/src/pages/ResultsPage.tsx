import { useMemo, useState } from 'react';
import { PageMeta } from '../components/layout/PageMeta';
import { PageHeader } from '../components/layout/PageHeader';
import { ButtonLink } from '../components/ui/ButtonLink';
import { Chip } from '../components/ui/Chip';
import { SectionHeading } from '../components/ui/SectionHeading';
import { FileCard } from '../components/workspace/FileCard';
import { MatchExplorer } from '../components/workspace/MatchExplorer';
import { PreviewDialog } from '../components/workspace/PreviewDialog';
import { ResultSummary } from '../components/workspace/ResultSummary';
import { ValidatorPanel } from '../components/workspace/ValidatorPanel';
import { pairTable, type ExportTable } from '../lib/export/rows';
import { useWorkspace } from '../lib/workspace/store';

export function ResultsPage() {
  const { result, restoring } = useWorkspace();
  const [preview, setPreview] = useState<ExportTable | null>(null);
  const tables = useMemo(() => (result ? { candidates: pairTable(result, 'candidates'), matching: pairTable(result, 'matching') } : null), [result]);

  if (restoring) {
    return (
      <>
        <PageMeta title="Results" />
        <PageHeader eyebrow="Workspace" title="Results" lead="Restoring your last results…" />
        <div className="container-page grid gap-4 md:grid-cols-4" aria-busy="true">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="h-28 animate-pulse rounded-card bg-surface/60" />
          ))}
        </div>
      </>
    );
  }

  if (!result || !tables) {
    return (
      <>
        <PageMeta title="Results" />
        <PageHeader eyebrow="Workspace" title="No results yet" lead="Results are kept for this browser tab only. Load the sample data and press Refine to see them." />
        <div className="container-page">
          <ButtonLink to="/app">Go to the workspace</ButtonLink>
        </div>
      </>
    );
  }

  return (
    <>
      <PageMeta title="Results" />
      <PageHeader eyebrow="Workspace" title="Results" />
      <div className="container-page -mt-6 space-y-6">
        <div className="flex flex-wrap items-center gap-3">
          <Chip tone="unsure">Demo mode: precomputed results</Chip>
          <p className="text-sm text-muted">
            Replayed from our real submission. The sample over-represents unmatched businesses and businesses with many matches, so its
            averages differ from the full test set.
          </p>
        </div>

        <ResultSummary summary={result.summary} />

        <div className="relative z-10 grid gap-4 md:grid-cols-2">
          <FileCard
            table={tables.candidates}
            idRows={result.candidates}
            description="Every record the pipeline considered for each business after the re-ranker cut, with whether it was matched."
            onPreview={() => setPreview(tables.candidates)}
          />
          <FileCard
            table={tables.matching}
            idRows={result.matches}
            description="The final matches: each record assigned to at most one business, kept only when the model was confident."
            onPreview={() => setPreview(tables.matching)}
          />
        </div>

        <ValidatorPanel result={result} />
      </div>

      <section className="container-page pt-20" aria-labelledby="explorer">
        <SectionHeading
          id="explorer"
          eyebrow="Match explorer"
          title="Explore one business"
          lead="Choose a business to see its records from Sources 2 and 3 side by side: the ones it was matched to, and the candidates it wasn’t."
        />
        <div className="mt-10">
          <MatchExplorer result={result} />
        </div>
      </section>

      <PreviewDialog table={preview} onClose={() => setPreview(null)} />
    </>
  );
}
