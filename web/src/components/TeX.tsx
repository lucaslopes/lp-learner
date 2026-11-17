import katex from 'katex';
import { useMemo } from 'react';

/** Math notation rendered by KaTeX. `block` gives display math. */
export function TeX({ children, block = false }: { children: string; block?: boolean }) {
  const html = useMemo(
    () => katex.renderToString(children, { displayMode: block, throwOnError: false, strict: 'ignore' }),
    [children, block],
  );
  return block ? <div className="tex-block" dangerouslySetInnerHTML={{ __html: html }} /> : <span dangerouslySetInnerHTML={{ __html: html }} />;
}
