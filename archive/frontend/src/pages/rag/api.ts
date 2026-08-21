import { apiFetch } from '../../api/client';

export interface Source {
  doc_id: string | null;
  title: string | null;
  source: string | null;
  date: string | null;
  score: number | null;
  snippet: string;
}

export interface TimelineEvent {
  date: string;
  title: string;
  summary: string;
  category: string;
}

export interface RagStatus {
  vector_chunks: number;
  documents: number;
  by_source: Record<string, number>;
  documents_dir: string;
  llm: {
    reachable: boolean;
    model: string;
    model_pulled?: boolean;
    error?: string;
  };
}

export interface IngestReport {
  processed: number;
  skipped: number;
  chunks: number;
  files: Array<{
    doc_id: string;
    source: string;
    date: string | null;
    title: string;
    chunks: number;
    bytes: number;
  }>;
}

// --------------------------------------------------------------------
// Q&A (streaming + non-streaming)
// --------------------------------------------------------------------

/**
 * Stream a question through the RAG pipeline.
 *
 * Calls onSources once up-front with the retrieved passages, then onToken
 * for every LLM text fragment, and finally resolves.
 */
export async function streamQuery(
  question: string,
  opts: {
    k?: number;
    signal?: AbortSignal;
    onToken: (t: string) => void;
    onSources: (s: Source[]) => void;
  },
): Promise<void> {
  const res = await apiFetch('api/rag/query/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: '*/*',
    },
    body: JSON.stringify({ question, k: opts.k ?? 6, stream: true }),
    signal: opts.signal,
  });

  if (!res.ok || !res.body) {
    throw new Error(`Query failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';

  // SSE frames are separated by a blank line.
  const drain = () => {
    let idx: number;
    while ((idx = buf.indexOf('\n\n')) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      handleFrame(frame);
    }
  };

  const handleFrame = (frame: string) => {
    let event: string | null = null;
    const dataLines: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
    }
    const data = dataLines.join('\n');

    if (event === 'sources') {
      try {
        const parsed = JSON.parse(data);
        opts.onSources(parsed.sources ?? []);
      } catch {
        /* ignore malformed */
      }
    } else if (event === 'done') {
      /* nothing */
    } else if (event === 'error') {
      try {
        const parsed = JSON.parse(data);
        throw new Error(parsed.error || 'LLM error');
      } catch (e) {
        if (e instanceof Error) throw e;
      }
    } else {
      // default event → token stream
      opts.onToken(data);
    }
  };

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { done, value } = await reader.read();
    if (value) {
      buf += decoder.decode(value, { stream: true });
      drain();
    }
    if (done) break;
  }
  // final flush in case trailing newline was missing
  buf += decoder.decode();
  drain();
}

export async function query(question: string, k = 6) {
  const res = await apiFetch('api/rag/query/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, k }),
  });
  if (!res.ok) throw new Error(`Query failed: ${res.status}`);
  return (await res.json()) as { answer: string; sources: Source[] };
}

// --------------------------------------------------------------------
// README / Timeline / Patterns
// --------------------------------------------------------------------

export async function generateReadme(topic?: string) {
  const res = await apiFetch('api/rag/generate-readme/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ topic: topic ?? '' }),
  });
  if (!res.ok) throw new Error(`README generation failed: ${res.status}`);
  return (await res.json()) as { readme: string; sources: Source[]; topic: string | null };
}

export async function getTimeline() {
  const res = await apiFetch('api/rag/timeline/');
  if (!res.ok) throw new Error(`Timeline failed: ${res.status}`);
  return (await res.json()) as {
    events: TimelineEvent[];
    document_count: number;
    total_chunks: number;
  };
}

export async function getPatterns(focus?: string) {
  const res = await apiFetch('api/rag/patterns/', {
    method: focus ? 'POST' : 'GET',
    headers: focus ? { 'Content-Type': 'application/json' } : undefined,
    body: focus ? JSON.stringify({ focus }) : undefined,
  });
  if (!res.ok) throw new Error(`Pattern analysis failed: ${res.status}`);
  return (await res.json()) as { report: string; focus: string | null; sources: Source[] };
}

// --------------------------------------------------------------------
// FAQ
// --------------------------------------------------------------------

export interface FaqPair {
  id: string;
  question: string;
  answer: string;
  annotation: string;
}

export async function getFaq() {
  const res = await apiFetch('api/rag/faq/');
  if (!res.ok) throw new Error(`FAQ fetch failed: ${res.status}`);
  return (await res.json()) as FaqPair[];
}

export async function saveFaqPair(pair: Omit<FaqPair, 'id'>) {
  const res = await apiFetch('api/rag/faq/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(pair),
  });
  if (!res.ok) throw new Error(`FAQ save failed: ${res.status}`);
  return (await res.json()) as FaqPair;
}

export async function deleteFaqPair(id: string) {
  const res = await apiFetch(`api/rag/faq/${id}/`, { method: 'DELETE' });
  if (!res.ok) throw new Error(`FAQ delete failed: ${res.status}`);
}

// --------------------------------------------------------------------
// Ingestion
// --------------------------------------------------------------------

export async function getStatus() {
  const res = await apiFetch('api/rag/status/');
  if (!res.ok) throw new Error(`Status failed: ${res.status}`);
  return (await res.json()) as RagStatus;
}

export async function rescan(force = false) {
  const res = await apiFetch('api/rag/ingest/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ force }),
  });
  if (!res.ok) throw new Error(`Ingest failed: ${res.status}`);
  return (await res.json()) as IngestReport;
}

export async function uploadFiles(files: File[]) {
  const form = new FormData();
  for (const f of files) form.append('file', f);

  const res = await apiFetch('api/rag/ingest/', {
    method: 'POST',
    body: form,
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return (await res.json()) as IngestReport;
}

