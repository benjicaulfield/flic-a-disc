import { useRef, useState } from 'react';
import type { FormEvent } from 'react';
import RagLayout from './RagLayout';
import { streamQuery, saveFaqPair } from './api';
import type { Source } from './api';

export default function RagDev() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [annotation, setAnnotation] = useState('');
  const [sources, setSources] = useState<Source[]>([]);
  const [busy, setBusy] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const ask = async (e: FormEvent) => {
    e.preventDefault();
    const q = question.trim();
    if (!q || busy) return;

    setAnswer('');
    setSources([]);
    setError(null);
    setSaved(false);
    setBusy(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamQuery(q, {
        signal: controller.signal,
        onSources: setSources,
        onToken: (tok) => setAnswer((a) => a + tok),
      });
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  };

  const stop = () => abortRef.current?.abort();

  const save = async () => {
    if (!question.trim() || !answer.trim()) return;
    setSaving(true);
    try {
      await saveFaqPair({ question: question.trim(), answer: answer.trim(), annotation: annotation.trim() });
      setSaved(true);
      setAnnotation('');
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  const discard = () => {
    setAnswer('');
    setSources([]);
    setAnnotation('');
    setError(null);
    setSaved(false);
  };

  return (
    <RagLayout>
      <div className="max-w-4xl mx-auto space-y-4">
        <form onSubmit={ask} className="flex gap-2">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about the project…"
            disabled={busy}
            className="flex-1 px-4 py-3 bg-[#252526] border border-[#3e3e42] rounded text-sm focus:outline-none focus:border-[#007acc] disabled:opacity-50"
          />
          {busy ? (
            <button
              type="button"
              onClick={stop}
              className="px-6 py-3 bg-[#5a1d1d] text-[#f48771] rounded text-xs hover:bg-[#6a2d2d]"
            >
              stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={!question.trim()}
              className="px-6 py-3 bg-[#007acc] text-white rounded text-xs hover:bg-[#1a8ad8] disabled:opacity-50"
            >
              ask
            </button>
          )}
        </form>

        {error && (
          <div className="bg-[#5a1d1d] border border-[#f48771] rounded p-3 text-xs text-[#f48771]">
            {error}
          </div>
        )}

        {(answer || busy) && (
          <div className="space-y-3">
            <div className="bg-[#252526] border border-[#3e3e42] rounded p-4">
              <div className="text-xs uppercase tracking-wider text-[#858585] mb-2">answer</div>
              <textarea
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                rows={10}
                className="w-full bg-transparent text-sm text-[#d4d4d4] resize-y focus:outline-none"
              />
              {busy && (
                <span className="inline-block w-2 h-4 bg-[#007acc] animate-pulse align-middle" />
              )}
            </div>

            {sources.length > 0 && (
              <details className="bg-[#252526] border border-[#3e3e42] rounded p-4">
                <summary className="text-xs text-[#858585] cursor-pointer hover:text-[#d4d4d4]">
                  {sources.length} source{sources.length !== 1 ? 's' : ''}
                </summary>
                <div className="mt-2 space-y-2">
                  {sources.map((s, i) => (
                    <div key={i} className="text-xs bg-[#1e1e1e] rounded p-2 border border-[#3e3e42]">
                      <div className="text-[#4ec9b0]">
                        {s.title || s.doc_id}
                        {s.date && <span className="text-[#858585]"> · {s.date}</span>}
                        {s.score != null && (
                          <span className="text-[#858585]"> · {(s.score * 100).toFixed(0)}%</span>
                        )}
                      </div>
                      <div className="text-[#858585] mt-1 line-clamp-2">{s.snippet}</div>
                    </div>
                  ))}
                </div>
              </details>
            )}

            {!busy && (
              <div className="space-y-2">
                <textarea
                  value={annotation}
                  onChange={(e) => setAnnotation(e.target.value)}
                  placeholder="Annotation / corrections (optional)…"
                  rows={3}
                  className="w-full px-4 py-3 bg-[#252526] border border-[#3e3e42] rounded text-sm text-[#d4d4d4] focus:outline-none focus:border-[#007acc] resize-y"
                />
                <div className="flex gap-2 justify-end">
                  <button
                    onClick={discard}
                    className="px-4 py-2 text-xs text-[#858585] hover:text-[#d4d4d4]"
                  >
                    discard
                  </button>
                  <button
                    onClick={save}
                    disabled={saving || saved}
                    className="px-6 py-2 bg-[#4ec9b0] text-[#1e1e1e] rounded text-xs hover:bg-[#6ed9c3] disabled:opacity-50"
                  >
                    {saved ? 'saved ✓' : saving ? 'saving…' : 'save to faq'}
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </RagLayout>
  );
}
