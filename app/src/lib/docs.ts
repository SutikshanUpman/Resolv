/** Anchor id for a numbered section of the documentation, e.g. '3.2' → 's-3-2', 'B' → 's-b'. */
export function sectionAnchor(section: string): string {
  return `s-${section.toLowerCase().replace(/\./g, '-')}`;
}

/** Anchor id for any heading: numbered headings use sectionAnchor, the rest a slug of their text. */
export function headingAnchor(text: string): string {
  const numbered = /^(\d+(?:\.\d+)*|[A-Z])\.?\s/.exec(text);
  if (numbered?.[1]) return sectionAnchor(numbered[1]);
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, '-')
    .replace(/^-|-$/g, '');
}

export interface TocEntry {
  level: 2 | 3;
  text: string;
  id: string;
}

/** The table of contents: every ## and ### heading outside code blocks. */
export function tableOfContents(markdown: string): TocEntry[] {
  const entries: TocEntry[] = [];
  let inCode = false;
  for (const line of markdown.split(/\r?\n/)) {
    if (line.startsWith('```')) inCode = !inCode;
    if (inCode) continue;
    const m = /^(#{2,3})\s+(.+?)\s*$/.exec(line);
    if (m?.[1] && m[2]) {
      const text = m[2].replace(/[*`]/g, '');
      entries.push({ level: m[1].length as 2 | 3, text, id: headingAnchor(text) });
    }
  }
  return entries;
}
