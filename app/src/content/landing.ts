// Landing-page copy. Numbers shown on the landing page come from metrics.ts and dataset.ts.

export const steps = [
  {
    title: 'Upload',
    text: 'Drop in your reference list of businesses (Source 1) and two more sources, as CSV, TSV or Excel. Each file is checked against the expected columns before anything runs.',
  },
  {
    title: 'Refine',
    text: 'Ten stages clean the names, find likely candidates, score every pair with learned models, and give each record to at most one business.',
  },
  {
    title: 'Download',
    text: 'Get two clean files back: every candidate considered, and the final matches. Preview them in the browser, or download as CSV, Excel, PDF or Word.',
  },
] as const;
