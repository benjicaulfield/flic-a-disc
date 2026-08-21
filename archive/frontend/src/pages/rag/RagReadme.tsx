import { useState } from 'react';
import type { FormEvent } from 'react';
import RagLayout from './RagLayout';
import { generateReadme } from './api';
import type { Source } from './api';
import { renderMarkdown } from './markdown';

export default function RagReadme() {
  const [topic, setTopic] = useState('');
  const [readme, setReadme] = useState('');
  const [sources, setSources] = useState<Source[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<'preview' | 'raw'>('preview');

  const generate = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setReadme('');
    try {
      const res = await generateReadme(topic.trim() || undefined);
      setReadme(res.readme);
      setSources(res.sources);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const download = () => {
    const blob = new Blob([readme], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'README.md';
    a.click();
    URL.revokeObjectURL(url);
  };

  const copy = async () => {
    await navigator.clipboard.writeText(readme);
  };

  return (
    <RagLayout>
      <div className="max-w-6xl mx-auto space-y-4">
        <form onSubmit={generate} className="flex gap-2">
          <input
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="Optional focus (e.g. 'deployment', 'eBay integration') — leave blank for full project"
            disabled={busy}
            className="flex-1 px-4 py-3 bg-[#252526] border border-[#3e3e42] rounded text-sm focus:outline-none focus:border-[#007acc] disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={busy}
            className="px-6 py-3 bg-[#007acc] text-white rounded text-xs hover:bg-[#1a8ad8] disabled:opacity-50"
          >
            {busy ? 'generating…' : 'generate'}
          </button>
        </form>

        {error && (
          <div className="bg-[#5a1d1d] border border-[#f48771] rounded p-3 text-xs text-[#f48771]">
            {error}
          </div>
        )}

        {readme && (
          <>
            <div className="flex items-center justify-between border-b border-[#3e3e42] pb-2">
              <div className="flex gap-2">
                <button
                  onClick={() => setView('preview')}
                  className={`px-3 py-1 text-xs rounded ${
                    view === 'preview' ? 'bg-[#37373d] text-white' : 'text-[#858585] hover:bg-[#2a2d2e]'
                  }`}
                >
                  preview
                </button>
                <button
                  onClick={() => setView('raw')}
                  className={`px-3 py-1 text-xs rounded ${
                    view === 'raw' ? 'bg-[#37373d] text-white' : 'text-[#858585] hover:bg-[#2a2d2e]'
                  }`}
                >
                  raw
                </button>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={copy}
                  className="px-3 py-1 text-xs text-[#858585] hover:text-[#d4d4d4]"
                >
                  copy
                </button>
                <button
                  onClick={download}
                  className="px-3 py-1 text-xs bg-[#4ec9b0] text-[#1e1e1e] rounded hover:bg-[#6ed9c3]"
                >
                  download .md
                </button>
              </div>
            </div>

            <div className="bg-[#252526] border border-[#3e3e42] rounded p-6">
              {view === 'preview' ? (
                <div
                  className="text-sm"
                  dangerouslySetInnerHTML={{ __html: renderMarkdown(readme) }}
                />
              ) : (
                <pre className="text-xs text-[#d4d4d4] whitespace-pre-wrap font-mono">
                  {readme}
                </pre>
              )}
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
