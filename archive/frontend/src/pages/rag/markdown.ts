/**
 * Minimal markdown → HTML renderer.
 *
 * This covers the subset of GFM that the RAG endpoints produce (headings,
 * bold/italic, fenced code blocks, inline code, bullet lists, paragraphs).
 * We deliberately avoid pulling in a full markdown library to keep the
 * frontend footprint small; the LLM output is trusted enough for a
 * read-only preview.
 */

function escapeHtml(s: string) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function inline(s: string) {
  return escapeHtml(s)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code class="px-1 bg-[#1e1e1e] rounded text-[#ce9178]">$1</code>')
    .replace(
      /\[([^\]]+)\]\(([^)]+)\)/g,
      '<a href="$2" class="text-[#4ec9b0] underline" target="_blank" rel="noreferrer">$1</a>',
    );
}

export function renderMarkdown(md: string): string {
  const lines = md.split('\n');
  const out: string[] = [];
  let i = 0;

  const flushList = (buf: string[], ordered: boolean) => {
    if (buf.length === 0) return;
    const tag = ordered ? 'ol' : 'ul';
    out.push(
      `<${tag} class="${ordered ? 'list-decimal' : 'list-disc'} pl-6 space-y-1 my-2">` +
        buf.map((item) => `<li>${inline(item)}</li>`).join('') +
        `</${tag}>`,
    );
  };

  while (i < lines.length) {
    const line = lines[i];

    // fenced code block
    if (line.startsWith('```')) {
      const lang = line.slice(3).trim();
      const body: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith('```')) {
        body.push(lines[i]);
        i++;
      }
      i++; // consume closing fence
      out.push(
        `<pre class="my-3 p-3 bg-[#1e1e1e] border border-[#3e3e42] rounded overflow-x-auto"><code class="text-[#ce9178] text-xs"${
          lang ? ` data-lang="${lang}"` : ''
        }>${escapeHtml(body.join('\n'))}</code></pre>`,
      );
      continue;
    }

    // heading
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      const level = h[1].length;
      const sizes = ['text-2xl', 'text-xl', 'text-lg', 'text-base', 'text-sm', 'text-xs'];
      out.push(
        `<h${level} class="${sizes[level - 1]} font-semibold text-white mt-4 mb-2">${inline(h[2])}</h${level}>`,
      );
      i++;
      continue;
    }

    // bullet / numbered list
    if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
      const ordered = /^\s*\d+\.\s+/.test(line);
      const buf: string[] = [];
      while (i < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[i])) {
        buf.push(lines[i].replace(/^\s*([-*]|\d+\.)\s+/, ''));
        i++;
      }
      flushList(buf, ordered);
      continue;
    }

    // blank line
    if (line.trim() === '') {
      i++;
      continue;
    }

    // paragraph: collect until blank line / special
    const para: string[] = [line];
    i++;
    while (
      i < lines.length &&
      lines[i].trim() !== '' &&
      !lines[i].startsWith('```') &&
      !/^(#{1,6})\s/.test(lines[i]) &&
      !/^\s*([-*]|\d+\.)\s+/.test(lines[i])
    ) {
      para.push(lines[i]);
      i++;
    }
    out.push(`<p class="my-2 leading-relaxed">${inline(para.join(' '))}</p>`);
  }

  return out.join('');
}
