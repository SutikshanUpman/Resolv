import { brand } from '../../content/brand';

interface PageMetaProps {
  /** Page name; the product name is appended. Omit on the home page. */
  title?: string;
  description?: string;
}

/** React 19 hoists <title> and <meta> into the document head. */
export function PageMeta({ title, description = brand.description }: PageMetaProps) {
  return (
    <>
      <title>{title ? `${title} · ${brand.name}` : `${brand.name}: ${brand.tagline}`}</title>
      <meta name="description" content={description} />
    </>
  );
}
