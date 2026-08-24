import { useState } from 'react';
import { DataTable } from '@/components/DataTable/DataTable';
import { AnnotationCell } from '@/components/AnnotationCell';
import type { ColumnDef } from '@tanstack/react-table';
import { apiFetch } from '../api/client';
import type { EbayListing } from '../types/ebay';
import { joinList } from '../features/deck/utils';

export const baseEbayColumns: ColumnDef<EbayListing>[] = [
  { accessorKey: 'ebay_title',      header: 'eBay Title',  size: 500 },
  { accessorKey: 'artist',          header: 'Artist',      size: 200 },
  { accessorKey: 'title',           header: 'Title',       size: 220 },
  { accessorKey: 'label',           header: 'Label',       size: 160 },
  { accessorKey: 'year',            header: 'Year',        size: 60,
    cell: ({ getValue }) => getValue<string>() ?? 'N/A' },
  { accessorKey: 'genres',          header: 'Genre',       size: 120,
    cell: ({ getValue }) => joinList(getValue()) },
  { accessorKey: 'styles',          header: 'Style',       size: 120,
    cell: ({ getValue }) => joinList(getValue()) },
  { accessorKey: 'media_condition', header: 'Cond.',       size: 80 },
  { accessorKey: 'keeper_score',    header: 'Score',       size: 70,
    cell: ({ getValue }) => getValue<number>()?.toFixed(3) ?? '—' },
];

interface EbayTabProps {
  isActive: boolean;
  endpoint: string;
  refreshEndpoint: string;
  columns: ColumnDef<EbayListing>[];
  title: string;
  storageKey: string;
}

export function EbayTab({ isActive: _isActive, endpoint, refreshEndpoint, columns, title, storageKey }: EbayTabProps) {
  const [allResults, setAllResults] = useState<EbayListing[]>(() => {
    try {
      const raw = sessionStorage.getItem(storageKey);
      return raw ? JSON.parse(raw).results ?? [] : [];
    } catch {
      return [];
    }
  });
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [annotating, setAnnotating] = useState(false);
  const [annotations, setAnnotations] = useState<Record<string, boolean>>({});
  const [submitting, setSubmitting] = useState(false);
  const [submitMessage, setSubmitMessage] = useState<string | null>(null);

  const toggleAnnotation = (ebayId: string, dbValue: boolean | null) => {
    setAnnotations(prev => {
      const displayed = prev[ebayId] ?? dbValue ?? false;
      return { ...prev, [ebayId]: !displayed };
    });
  };

  const loadListings = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiFetch(endpoint);
      if (!response.ok) throw new Error('request failed');
      const json = await response.json();
      const results = json.listings ?? [];
      setAllResults(results);
      sessionStorage.setItem(storageKey, JSON.stringify({ results }));
    } catch {
      setError('Failed to load listings');
    } finally {
      setLoading(false);
    }
  };

  const refresh = async () => {
    setRefreshing(true);
    setError(null);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 25 * 60 * 1000);
    try {
      const response = await apiFetch(refreshEndpoint, {
        method: 'POST',
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (!response.ok) throw new Error('refresh failed');
      await loadListings();
    } catch (err) {
      clearTimeout(timeoutId);
      if (err instanceof Error && err.name === 'AbortError') {
        setError('Request timed out.');
      } else {
        setError('Refresh failed.');
      }
    } finally {
      setRefreshing(false);
    }
  };

  const submitAnnotations = async () => {
    if (allResults.length === 0) return;

    setSubmitting(true);
    setSubmitMessage(null);

    try {
      const response = await apiFetch('api/ebay/annotate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          // Every row shown gets a decision — explicit true/false if toggled,
          // implicit false otherwise. An unclicked row means "reviewed, not
          // a match," not "skip this one" — same rule BySeller enforces.
          annotations: allResults.map(r => ({
            ebay_id: r.ebay_id,
            label: annotations[r.ebay_id] ?? false,
          })),
        }),
      });

      const json = await response.json();
      if (!response.ok) throw new Error('Request failed');

      setAnnotations({});
      const errors: string[] = json.errors ?? [];
      setSubmitMessage(
        `Saved ${json.keepers ?? 0} keeper(s), ${json.non_keepers ?? 0} non-keeper(s).` +
        (errors.length ? ` ${errors.length} error(s): ${errors.join('; ')}` : ''),
      );
    } catch {
      setSubmitMessage('Failed to submit annotations.');
    } finally {
      setSubmitting(false);
    }
  };

  const displayColumns = annotating
    ? [
        ...columns,
        {
          id: 'annotate',
          header: 'Keeper',
          size: 60,
          cell: ({ row }: { row: { original: EbayListing } }) => (
            <AnnotationCell
              dbValue={row.original.wanted}
              dbEvaluated={row.original.evaluated}
              override={annotations[row.original.ebay_id]}
              onToggle={() => toggleAnnotation(row.original.ebay_id, row.original.wanted)}
            />
          ),
        },
      ]
    : columns;

  return (
    <div className="w-full px-6 py-6">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-3xl font-bold text-slate-700">{title}</h1>
        <button
          onClick={() => setAnnotating(a => !a)}
          className={`px-4 py-2 text-sm rounded border ${
            annotating
              ? 'bg-blue-600 text-white border-blue-600'
              : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
          }`}
        >
          {annotating ? 'Annotating' : 'Annotate'}
        </button>
        {annotating && (
          <button
            onClick={submitAnnotations}
            disabled={submitting || Object.keys(annotations).length === 0}
            className="px-4 py-2 text-sm rounded border bg-blue-600 text-white border-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            {submitting ? 'Submitting...' : 'Submit Annotations'}
          </button>
        )}
        <button
          onClick={refresh}
          disabled={refreshing}
          className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:bg-gray-400 text-sm"
        >
          {refreshing ? 'Refreshing...' : '🔄 Refresh'}
        </button>
      </div>

      {error && <div className="mb-4 text-red-500 text-sm">{error}</div>}
      {submitMessage && <div className="mb-4 text-sm text-gray-700">{submitMessage}</div>}

      {loading ? (
        <p className="p-4 text-gray-500">Loading...</p>
      ) : (
        <DataTable data={allResults} columns={displayColumns} />
      )}
    </div>
  );
}