import { brand } from '../../content/brand';
import type { ExportTable } from './rows';

export type ExportFormat = 'csv' | 'xlsx' | 'pdf' | 'docx' | 'tsv';

/** PDF and Word tables are capped; CSV and Excel always hold every row. */
export const DOCUMENT_ROW_CAP = 500;

const DEMO_NOTE = 'Demo mode: precomputed results from our real submission on a sample of the challenge test set.';

export function saveBlob(blob: Blob, fileName: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = fileName;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function csvCell(v: string): string {
  return /[",\n\r]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;
}

export function toCsv(t: ExportTable): Blob {
  const text = [t.columns, ...t.rows].map((r) => r.map(csvCell).join(',')).join('\r\n');
  // BOM so Excel opens Indic scripts and accents correctly.
  return new Blob(['﻿', text], { type: 'text/csv;charset=utf-8' });
}

export async function toXlsx(t: ExportTable): Promise<Blob> {
  const XLSX = await import('xlsx');
  const sheet = XLSX.utils.aoa_to_sheet([t.columns, ...t.rows]);
  sheet['!cols'] = t.columns.map((c) => ({ wch: c.includes('name') || c.includes('address') ? 40 : 18 }));
  const book = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(book, sheet, t.title.slice(0, 31));
  XLSX.utils.book_append_sheet(book, XLSX.utils.aoa_to_sheet([[DEMO_NOTE]]), 'About');
  const out = XLSX.write(book, { bookType: 'xlsx', type: 'array', compression: true }) as ArrayBuffer;
  return new Blob([out], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
}

/** jsPDF's built-in fonts cover Latin-1 only, so the PDF keeps rows written entirely in Latin script. */
const LATIN = /^[\t\u0020-\u007E\u00A0-\u00FF\u2013\u2014\u2018-\u201E\u2026\u20AC\u0152\u0153]*$/;

export async function toPdf(t: ExportTable): Promise<Blob> {
  const [{ jsPDF }, { autoTable }] = await Promise.all([import('jspdf'), import('jspdf-autotable')]);
  const latin = t.rows.filter((r) => r.every((c) => LATIN.test(c)));
  const rows = latin.slice(0, DOCUMENT_ROW_CAP);
  const skippedScript = t.rows.length - latin.length;
  const doc = new jsPDF({ orientation: 'landscape', unit: 'pt', format: 'a4' });
  doc.setFontSize(14);
  doc.text(`${brand.name}: ${t.title}`, 40, 40);
  doc.setFontSize(8);
  const notes = [
    DEMO_NOTE,
    `This PDF shows ${rows.length.toLocaleString('en-US')} of ${t.rows.length.toLocaleString('en-US')} rows: at most ${DOCUMENT_ROW_CAP} rows, written in Latin script only.`,
    `${skippedScript.toLocaleString('en-US')} rows containing Indic scripts were left out because the PDF fonts cannot display them. Download CSV or Excel for every row.`,
  ];
  notes.forEach((n, i) => doc.text(n, 40, 58 + i * 11));
  autoTable(doc, {
    head: [t.columns],
    body: rows,
    startY: 100,
    styles: { fontSize: 6.5, cellPadding: 2, overflow: 'linebreak' },
    headStyles: { fillColor: [47, 107, 255] },
    margin: { left: 24, right: 24 },
  });
  return doc.output('blob');
}

export async function toDocx(t: ExportTable): Promise<Blob> {
  const { Document, Packer, Paragraph, Table, TableRow, TableCell, TextRun, WidthType, PageOrientation, HeadingLevel } = await import('docx');
  const rows = t.rows.slice(0, DOCUMENT_ROW_CAP);
  const cell = (text: string, bold = false) =>
    new TableCell({ children: [new Paragraph({ children: [new TextRun({ text, bold, size: 14 })] })] });
  const doc = new Document({
    sections: [
      {
        properties: { page: { size: { orientation: PageOrientation.LANDSCAPE } } },
        children: [
          new Paragraph({ text: `${brand.name}: ${t.title}`, heading: HeadingLevel.HEADING_1 }),
          new Paragraph({ children: [new TextRun({ text: DEMO_NOTE, size: 18 })] }),
          new Paragraph({
            children: [
              new TextRun({
                text: `This document shows the first ${rows.length.toLocaleString('en-US')} of ${t.rows.length.toLocaleString('en-US')} rows (capped at ${DOCUMENT_ROW_CAP}). Download CSV or Excel for every row.`,
                size: 18,
              }),
            ],
          }),
          new Table({
            width: { size: 100, type: WidthType.PERCENTAGE },
            rows: [new TableRow({ tableHeader: true, children: t.columns.map((c) => cell(c, true)) }), ...rows.map((r) => new TableRow({ children: r.map((c) => cell(c)) }))],
          }),
        ],
      },
    ],
  });
  return Packer.toBlob(doc);
}

export const formatLabels: Record<ExportFormat, string> = {
  csv: 'CSV',
  xlsx: 'Excel (.xlsx)',
  pdf: `PDF (first ${DOCUMENT_ROW_CAP} Latin-script rows)`,
  docx: `Word (.docx, first ${DOCUMENT_ROW_CAP} rows)`,
  tsv: 'Challenge format (.tsv)',
};
