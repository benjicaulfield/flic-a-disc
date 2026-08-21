import { useEffect, useState } from 'react';
import RagLayout from './RagLayout';
import { getFaq, deleteFaqPair } from './api';
import type { FaqPair } from './api';

export default function RagFAQ() {
  const [pairs, setPairs] = useState<FaqPair[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getFaq()
      .then(setPairs)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const remove = async (id: string) => {
    try {
      await deleteFaqPair(id);
      setPairs((p) => p.filter((x) => x.id !== id));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  if (loading) return (
    <RagLayout>
      <div className="text-[#858585] text-sm text-center py-20">loading…</div>
    </RagLayout>
  );

  return (
    <RagLayout>
      <div className="max-w-4xl mx-auto space-y-4">
        {error && (
          <div className="bg-[#5a1d1d] border border-[#f48771] rounded p-3 text-xs text-[#f48771]">
            {error}
          </div>
        )}

        {pairs.length === 0 ? (
          <div className="text-center py-20 text-[#6a6a6a] text-sm">
            <p>No FAQ pairs saved yet.</p>
            <p className="mt-2 text-xs">Use the <span className="text-[#d4d4d4]">dev</span> tab to ask questions and save answers.</p>
          </div>
        ) : (
          <>
            <div className="text-xs text-[#858585]">{pairs.length} pair{pairs.length !== 1 ? 's' : ''}</div>
            {pairs.map((p) => (
              <div key={p.id} className="bg-[#252526] border border-[#3e3e42] rounded p-5 space-y-3">
                <div className="flex items-start justify-between gap-4">
                  <div className="text-sm font-semibold text-white">{p.question}</div>
                  <button
                    onClick={() => remove(p.id)}
                    className="text-xs text-[#858585] hover:text-[#f48771] shrink-0"
                  >
                    delete
                  </button>
                </div>
                <div className="text-sm text-[#d4d4d4] whitespace-pre-wrap">{p.answer}</div>
                {p.annotation && (
                  <div className="text-xs text-[#dcdcaa] bg-[#2a2d2e] border border-[#3e3e42] rounded p-3 whitespace-pre-wrap">
                    <span className="text-[#858585] uppercase tracking-wider mr-2">note</span>
                    {p.annotation}
                  </div>
                )}
              </div>
            ))}
          </>
        )}
      </div>
    </RagLayout>
  );
}
