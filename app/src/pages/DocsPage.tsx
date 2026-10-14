import { Children, isValidElement, useEffect, type ReactNode } from 'react';
import Markdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useLocation } from 'react-router';
import documentation from '../../../code/Documentation_template.md?raw';
import { PageMeta } from '../components/layout/PageMeta';
import { headingAnchor, tableOfContents } from '../lib/docs';

const toc = tableOfContents(documentation);

function textOf(node: ReactNode): string {
  return Children.toArray(node)
    .map((c) => (typeof c === 'string' || typeof c === 'number' ? String(c) : isValidElement<{ children?: ReactNode }>(c) ? textOf(c.props.children) : ''))
    .join('');
}

function heading(Tag: 'h1' | 'h2' | 'h3' | 'h4', className: string): NonNullable<Components['h1']> {
  return function Heading({ children }) {
    const id = headingAnchor(textOf(children));
    return (
      <Tag id={id} className={`group ${className}`}>
        {children}
        {Tag !== 'h1' && (
          <a href={`#${id}`} aria-label="Link to this section" className="ml-2 text-glow opacity-0 transition-opacity group-hover:opacity-100 focus:opacity-100">
            #
          </a>
        )}
      </Tag>
    );
  };
}

const components: Components = {
  h1: heading('h1', 'text-h1 font-semibold text-balance'),
  h2: heading('h2', 'mt-16 border-t border-line pt-10 text-h2 font-semibold'),
  h3: heading('h3', 'mt-10 text-h3 font-semibold'),
  h4: heading('h4', 'mt-8 text-lg font-semibold'),
  p: ({ children }) => <p className="mt-4 leading-7 text-ink/90">{children}</p>,
  ul: ({ children }) => <ul className="mt-4 list-disc space-y-2 pl-6 text-ink/90 marker:text-glow">{children}</ul>,
  ol: ({ children }) => <ol className="mt-4 list-decimal space-y-2 pl-6 text-ink/90 marker:text-muted">{children}</ol>,
  a: ({ href, children }) => (
    <a href={href} className="text-glow underline underline-offset-4">
      {children}
    </a>
  ),
  strong: ({ children }) => <strong className="font-semibold text-ink">{children}</strong>,
  hr: () => <hr className="my-10 border-line" />,
  table: ({ children }) => (
    <div className="mt-6 overflow-x-auto rounded-xl border border-line">
      <table className="w-full min-w-[32rem] border-collapse text-sm">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-raised/70 text-left">{children}</thead>,
  th: ({ children }) => <th className="border-b border-line px-3 py-2 font-medium text-ink">{children}</th>,
  td: ({ children }) => <td className="border-t border-line px-3 py-2 align-top text-ink/85">{children}</td>,
  pre: ({ children }) => (
    <pre className="mt-6 overflow-x-auto rounded-xl border border-line bg-surface p-4 font-mono text-[0.8rem] leading-6 text-ink/90">
      {children}
    </pre>
  ),
  code: ({ className, children }) =>
    className ? (
      <code className={className}>{children}</code>
    ) : (
      <code className="rounded bg-raised px-1.5 py-0.5 font-mono text-[0.85em] [overflow-wrap:anywhere] text-glow">{children}</code>
    ),
};

function TocList() {
  return (
    <ol className="mt-4 space-y-1 text-sm">
      {toc.map((e) => (
        <li key={e.id} className={e.level === 3 ? 'pl-4' : ''}>
          <a href={`#${e.id}`} className={`block rounded px-2 py-1 hover:bg-raised/60 hover:text-ink ${e.level === 2 ? 'text-ink/90' : 'text-muted'}`}>
            {e.text}
          </a>
        </li>
      ))}
    </ol>
  );
}

export function DocsPage() {
  const { hash } = useLocation();

  // Jump to a section when arriving with #s-3-2 (links from the architecture page).
  useEffect(() => {
    if (!hash) return;
    const frame = requestAnimationFrame(() => document.getElementById(decodeURIComponent(hash.slice(1)))?.scrollIntoView());
    return () => cancelAnimationFrame(frame);
  }, [hash]);

  return (
    <>
      <PageMeta title="Documentation" description="Our full methodology write-up for the Amazon ML Challenge 2026." />
      <div className="container-page grid gap-12 pt-16 lg:grid-cols-[16rem_minmax(0,1fr)] lg:pt-20">
        <nav aria-label="Table of contents" className="lg:sticky lg:top-24 lg:max-h-[calc(100dvh-7rem)] lg:self-start lg:overflow-y-auto">
          <details className="rounded-card border border-line bg-surface/50 p-4 lg:hidden">
            <summary className="cursor-pointer text-xs font-medium tracking-[0.14em] text-glow uppercase">Contents</summary>
            <TocList />
          </details>
          <div className="hidden lg:block">
            <p className="text-xs font-medium tracking-[0.14em] text-glow uppercase">Contents</p>
            <TocList />
          </div>
        </nav>
        <article className="min-w-0 max-w-[52rem] pb-8">
          <Markdown remarkPlugins={[remarkGfm]} components={components}>
            {documentation}
          </Markdown>
        </article>
      </div>
    </>
  );
}
