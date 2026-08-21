import { useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { getStatus } from './api';
import type { RagStatus } from './api';

const tabs = [
  { path: '/rag/dev', label: 'dev' },
  { path: '/rag/faq', label: 'faq' },
  { path: '/rag/readme', label: 'readme' },
  { path: '/rag/ingest', label: 'ingest' },
];

export default function RagLayout({ children }: { children: ReactNode }) {
  const navigate = useNavigate();
  const [status, setStatus] = useState<RagStatus | null>(null);

  useEffect(() => {
    getStatus()
      .then(setStatus)
      .catch(() => setStatus(null));
  }, []);

  return (
    <div className="min-h-screen bg-[#1e1e1e] text-[#d4d4d4] font-mono flex flex-col">
      {/* header */}
      <header className="border-b border-[#3e3e42] bg-[#252526] px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <button
            onClick={() => navigate('/dashboard')}
            className="text-xs text-[#858585] hover:text-[#d4d4d4]"
          >
            ← dashboard
          </button>
          <h1 className="text-sm font-semibold text-white">knowledge base</h1>
        </div>
        <div className="text-xs text-[#858585] flex items-center gap-4">
          {status && (
            <>
              <span>
                {status.vector_chunks.toLocaleString()} chunks · {status.documents} docs
              </span>
              <span
                className={`h-2 w-2 rounded-full ${
                  status.llm.reachable && status.llm.model_pulled
                    ? 'bg-[#4ec9b0]'
                    : status.llm.reachable
                      ? 'bg-[#dcdcaa]'
                      : 'bg-[#f48771]'
                }`}
                title={
                  status.llm.reachable
                    ? status.llm.model_pulled
                      ? `ollama ready · ${status.llm.model}`
                      : `model ${status.llm.model} not pulled`
                    : 'ollama unreachable'
                }
              />
            </>
          )}
        </div>
      </header>

      {/* tabs */}
      <nav className="border-b border-[#3e3e42] bg-[#252526] px-6 flex gap-1">
        {tabs.map((t) => (
          <NavLink
            key={t.path}
            to={t.path}
            className={({ isActive }) =>
              `px-4 py-2 text-xs border-b-2 transition-colors ${
                isActive
                  ? 'border-[#007acc] text-white'
                  : 'border-transparent text-[#858585] hover:text-[#d4d4d4]'
              }`
            }
          >
            {t.label}
          </NavLink>
        ))}
      </nav>

      {/* body */}
      <main className="flex-1 overflow-y-auto p-6">{children}</main>
    </div>
  );
}
