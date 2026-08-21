import { useRef, useState } from 'react';
import type { FormEvent } from 'react';
import RagLayout from './RagLayout';
import { streamQuery } from './api';
import type { Source } from './api';

interface Message {
  role: 'user' | 'assistant';
  text: string;
  sources?: Source[];
}

export default function RagQuery() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const ask = async (e: FormEvent) => {
    e.preventDefault();
    const q = input.trim();
    if (!q || busy) return;

    setInput('');
    setError(null);
    setBusy(true);

    // push user + empty assistant placeholder
    setMessages((m) => [...m, { role: 'user', text: q }, { role: 'assistant', text: '' }]);
    const assistantIndex = messages.length + 1;

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamQuery(q, {
        signal: controller.signal,
        onSources: (srcs) => {
          setMessages((m) => {
            const copy = [...m];
            copy[assistantIndex] = { ...copy[assistantIndex], sources: srcs };
            return copy;
          });
        },
        onToken: (tok) => {
          setMessages((m) => {
            const copy = [...m];
            const curr = copy[assistantIndex];
            copy[assistantIndex] = { ...curr, text: curr.text + tok };
            return copy;
          });
        },
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

  const stop = () => {
    abortRef.current?.abort();
  };

  return (
    <RagLayout>
      <div className="max-w-5xl mx-auto flex flex-col h-full gap-4">
        {/* transcript */}
        <div className="flex-1 min-h-[400px] space-y-4 overflow-y-auto">
          {messages.length === 0 && (
            <div className="text-center py-20 text-[#6a6a6a] text-sm">
              <p>Ask anything about your development conversations.</p>
              <p className="mt-2 text-xs">
                try: "How was the eBay API integrated?" or "What database migration issues came up?"
              </p>
            </div>
          )}
          {messages.map((m, i) => (
            <div
              key={i}
              className={`rounded p-4 ${
                m.role === 'user'
                  ? 'bg-[#2a2d2e] border border-[#3e3e42]'
                  : 'bg-[#252526] border border-[#3e3e42]'
              }`}
            >
              <div className="text-xs uppercase tracking-wider text-[#858585] mb-2">
                {m.role === 'user' ? 'you' : 'assistant'}
              </div>
              <div className="text-sm whitespace-pre-wrap">
                {m.text}
                {m.role === 'assistant' && busy && i === messages.length - 1 && (
                  <span className="inline-block w-2 h-4 bg-[#007acc] animate-pulse ml-1 align-middle" />
                )}
              </div>
              {m.sources && m.sources.length > 0 && (
                <details className="mt-3">
                  <summary className="text-xs text-[#858585] cursor-pointer hover:text-[#d4d4d4]">
                    {m.sources.length} source{m.sources.length !== 1 ? 's' : ''}
                  </summary>
                  <div className="mt-2 space-y-2">
                    {m.sources.map((s, j) => (
                      <div key={j} className="text-xs bg-[#1e1e1e] rounded p-2 border border-[#3e3e42]">
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
            </div>
          ))}
        </div>

        {error && (
          <div className="bg-[#5a1d1d] border border-[#f48771] rounded p-3 text-xs text-[#f48771]">
            {error}
          </div>
        )}

        {/* input */}
        <form onSubmit={ask} className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question…"
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
              disabled={!input.trim()}
              className="px-6 py-3 bg-[#007acc] text-white rounded text-xs hover:bg-[#1a8ad8] disabled:opacity-50"
            >
              ask
            </button>
          )}
        </form>
      </div>
    </RagLayout>
  );
}
