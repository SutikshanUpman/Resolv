import { Link } from 'react-router';
import { CandidateFunnel } from '../components/architecture/CandidateFunnel';
import { ErrorBreakdown } from '../components/architecture/ErrorBreakdown';
import { F05Explainer } from '../components/architecture/F05Explainer';
import { Glossary } from '../components/architecture/Glossary';
import { PipelineFlow } from '../components/architecture/PipelineFlow';
import { PipelineOverview } from '../components/architecture/PipelineOverview';
import { VersionTimeline } from '../components/architecture/VersionTimeline';
import { PageHeader } from '../components/layout/PageHeader';
import { PageMeta } from '../components/layout/PageMeta';
import { ButtonLink } from '../components/ui/ButtonLink';
import { SectionHeading } from '../components/ui/SectionHeading';
import { overview } from '../content/overview';

export function ArchitecturePage() {
  return (
    <>
      <PageMeta title="How it works" description="The ten-stage entity resolution pipeline, its numbers, and where it still makes mistakes." />
      <PageHeader
        eyebrow="Architecture"
        title="How it works"
        lead={
          <>
            Our pipeline has ten stages, grouped into three phases. Every number on this page comes from our{' '}
            <Link to="/architecture/docs" className="text-glow hover:underline">
              documentation
            </Link>
             and says which data it was measured on.
          </>
        }
      />

      <section className="container-page py-12" aria-labelledby="overview">
        <SectionHeading id="overview" eyebrow="At a glance" title="The pipeline in three phases" lead={overview.lead} />
        <PipelineOverview />
      </section>

      <section className="container-page py-20" aria-labelledby="pipeline">
        <SectionHeading
          id="pipeline"
          eyebrow="Pipeline"
          title="All ten stages"
          lead="Click any stage on the map, or use the arrows to walk through them one by one."
        />
        <PipelineFlow />
      </section>

      <section className="container-page py-20" aria-labelledby="terms">
        <SectionHeading id="terms" eyebrow="Glossary" title="Terms used on this page" />
        <Glossary />
      </section>

      <section className="container-page py-20" aria-labelledby="funnel">
        <SectionHeading
          id="funnel"
          eyebrow="Search space"
          title="How the search space shrinks"
          lead="How many candidates each business has left after each step. Validation and test numbers are labelled separately."
        />
        <CandidateFunnel />
      </section>

      <section className="container-page py-20" aria-labelledby="versions">
        <SectionHeading
          id="versions"
          eyebrow="Version history"
          title="How our score improved"
          lead="Each point is a version we submitted. Click one to see what went wrong and how we fixed it in the next version."
        />
        <VersionTimeline />
      </section>

      <section className="container-page py-20" aria-labelledby="metric">
        <SectionHeading id="metric" eyebrow="The metric" title="How we were scored" />
        <F05Explainer />
      </section>

      <section className="container-page py-20" aria-labelledby="errors">
        <SectionHeading
          id="errors"
          eyebrow="Error analysis"
          title="Where it still makes mistakes"
          lead="Most of what we still lose is recall: records we chose not to match, because the text alone cannot tell two similar businesses apart."
        />
        <ErrorBreakdown />
      </section>

      <section className="container-page pt-8">
        <div className="glass flex flex-col items-start justify-between gap-6 rounded-card p-8 md:flex-row md:items-center">
          <div>
            <h2 className="text-h3 font-semibold">Read the full write-up</h2>
            <p className="mt-2 text-muted">Features, training details, ablations and the reproduction steps.</p>
          </div>
          <ButtonLink to="/architecture/docs" variant="secondary">
            Open the documentation
          </ButtonLink>
        </div>
      </section>
    </>
  );
}
