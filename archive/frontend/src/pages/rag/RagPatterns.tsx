import { useState } from 'react';
import type { FormEvent } from 'react';
import RagLayout from './RagLayout';
import { getPatterns } from './api';
import type { Source } from './api';
import { renderMarkdown } from './markdown';

export default function RagPatterns() {
  const [focus, setFocus] = useState('');
  const [report, setReport] = useState('');
  const [sources, setSources] = useState<Source[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async (e?: FormEvent) => {
    e?.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await getPatterns(focus.trim() || undefined);
      setReport(res.report);
      setSources(res.sources);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const download = () => {
    const blob = new Blob([report], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'patterns.md';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <RagLayout>
      <div className="max-w-5xl mx-auto space-y-4">
        <form onSubmit={run} className="flex gap-2">
          <input
            value={focus}
            onChange={(e) => setFocus(e.target.value)}
            placeholder="Optional focus area (e.g. 'error handling', 'testing') — leave blank for general retro"
            disabled={busy}
            className="flex-1 px-4 py-3 bg-[#252526] border border-[#3e3e42] rounded text-sm focus:outline-none focus:border-[#007acc] disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={busy}
            className="px-6 py-3 bg-[#007acc] text-white rounded text-xs hover:bg-[#1a8ad8] disabled:opacity-50"
          >
            {busy ? 'analysing…' : 'analyse'}
          </button>
        </form>

        {error && (
          <div className="bg-[#5a1d1d] border border-[#f48771] rounded p-3 text-xs text-[#f48771]">
            {error}
          </div>
        )}

        {report && (
          <>
            <div className="flex justify-end">
              <button
                onClick={download}
                className="px-3 py-1 text-xs bg-[#4ec9b0] text-[#1e1e1e] rounded hover:bg-[#6ed9c3]"
              >
                download .md
              </button>
            </div>
            <div className="bg-[#252526] border border-[#3e3e42] rounded p-6">
              <div className="text-sm" dangerouslySetInnerHTML={{ __html: renderMarkdown(report) }} />
            </div>

            {sources.length > 0 && (
              <details className="bg-[#252526] border border-[#3e3e42] rounded p-4">
                <summary className="text-xs text-[#858585] cursor-pointer">
                  derived from {sources.length} sources
                </summary>
                <div className="mt-3 grid grid-cols-2 gap-2">
                  {sources.map((s, i) => (
                    <div key={i} className="text-xs bg-[#1e1e1e] rounded p-2">
                      <div className="text-[#4ec9b0]">{s.title || s.doc_id}</div>
                      <div className="text-[#858585]">
                        {s.source} {s.date && `· ${s.date}`}
                      </div>
                    </div>
                  ))}
                </div>
              </details>
            )}
          </>
        )}
      </div>
    </RagLayout>
  );
}
