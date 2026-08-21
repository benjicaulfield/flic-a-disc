import { useEffect, useState } from 'react';
import type { ChangeEvent } from 'react';
import RagLayout from './RagLayout';
import { getStatus, rescan, uploadFiles } from './api';
import type { IngestReport, RagStatus } from './api';

export default function RagIngest() {
  const [status, setStatus] = useState<RagStatus | null>(null);
  const [report, setReport] = useState<IngestReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    try {
      setStatus(await getStatus());
    } catch {
      /* already shown in layout */
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const doRescan = async (force: boolean) => {
    setBusy(true);
    setError(null);
    setReport(null);
    try {
      const r = await rescan(force);
      setReport(r);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const onFiles = async (e: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    if (files.length === 0) return;
    setBusy(true);
    setError(null);
    setReport(null);
    try {
      const r = await uploadFiles(files);
      setReport(r);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
      e.target.value = '';
    }
  };

  return (
    <RagLayout>
      <div className="max-w-5xl mx-auto space-y-6">
        {/* status card */}
        {status && (
          <div className="bg-[#252526] border border-[#3e3e42] rounded p-5 grid grid-cols-3 gap-4 text-sm">
            <div>
              <div className="text-xs text-[#858585] uppercase mb-1">chunks</div>
              <div className="text-2xl text-white">{status.vector_chunks.toLocaleString()}</div>
            </div>
            <div>
              <div className="text-xs text-[#858585] uppercase mb-1">documents</div>
              <div className="text-2xl text-white">{status.documents}</div>
            </div>
            <div>
              <div className="text-xs text-[#858585] uppercase mb-1">llm</div>
              <div
                className={`text-sm ${
                  status.llm.reachable && status.llm.model_pulled
                    ? 'text-[#4ec9b0]'
                    : status.llm.reachable
                      ? 'text-[#dcdcaa]'
                      : 'text-[#f48771]'
                }`}
              >
                {status.llm.model}
                {!status.llm.reachable && ' (offline)'}
                {status.llm.reachable && !status.llm.model_pulled && ' (not pulled)'}
              </div>
            </div>
            <div className="col-span-3 text-xs text-[#858585]">
              corpus: <code className="text-[#ce9178]">{status.documents_dir}</code>
              {Object.keys(status.by_source).length > 0 && (
                <span className="ml-4">
                  {Object.entries(status.by_source)
                    .map(([k, v]) => `${k}: ${v}`)
                    .join(' · ')}
                </span>
              )}
            </div>
          </div>
        )}

        {/* actions */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-[#252526] border border-[#3e3e42] rounded p-5">
            <h3 className="text-sm font-semibold text-white mb-2">rescan corpus</h3>
            <p className="text-xs text-[#858585] mb-4">
              Walk <code className="text-[#ce9178]">{status?.documents_dir}</code> and embed any new
              or changed files. Unchanged files are skipped.
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => doRescan(false)}
                disabled={busy}
                className="px-4 py-2 bg-[#007acc] text-white rounded text-xs hover:bg-[#1a8ad8] disabled:opacity-50"
              >
                incremental scan
              </button>
              <button
                onClick={() => doRescan(true)}
                disabled={busy}
                className="px-4 py-2 bg-[#37373d] text-[#d4d4d4] rounded text-xs hover:bg-[#4a4a4d] disabled:opacity-50"
                title="Re-embed every file regardless of hash"
              >
                force rebuild
              </button>
            </div>
          </div>

          <div className="bg-[#252526] border border-[#3e3e42] rounded p-5">
            <h3 className="text-sm font-semibold text-white mb-2">upload new conversations</h3>
            <p className="text-xs text-[#858585] mb-4">
              Drop additional .txt transcripts here. They are saved to{' '}
              <code className="text-[#ce9178]">uploads/</code> and embedded immediately.
            </p>
            <label className="block">
              <input
                type="file"
                accept=".txt"
                multiple
                disabled={busy}
                onChange={onFiles}
                className="hidden"
              />
              <span className="inline-block px-4 py-2 bg-[#4ec9b0] text-[#1e1e1e] rounded text-xs hover:bg-[#6ed9c3] cursor-pointer">
                choose files…
              </span>
            </label>
          </div>
        </div>

        {busy && (
          <div className="flex items-center gap-3 text-xs text-[#858585]">
            <div className="animate-spin rounded-full h-4 w-4 border-2 border-[#007acc] border-t-transparent" />
            embedding…
          </div>
        )}

        {error && (
          <div className="bg-[#5a1d1d] border border-[#f48771] rounded p-3 text-xs text-[#f48771]">
            {error}
          </div>
        )}

        {/* result */}
        {report && (
          <div className="bg-[#252526] border border-[#3e3e42] rounded p-5">
            <div className="text-xs text-[#858585] mb-3">
              processed {report.processed} · skipped {report.skipped} · {report.chunks} chunks
            </div>
            {report.files.length > 0 && (
              <div className="space-y-1 max-h-80 overflow-y-auto">
                {report.files.map((f, i) => (
                  <div key={i} className="text-xs bg-[#1e1e1e] rounded p-2 flex justify-between">
                    <div>
                      <span className="text-[#4ec9b0]">{f.title || f.doc_id}</span>
                      <span className="text-[#858585] ml-2">
                        [{f.source}] {f.date || ''}
                      </span>
                    </div>
                    <span className="text-[#858585]">{f.chunks} chunks</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </RagLayout>
  );
}
