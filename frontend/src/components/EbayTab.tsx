import { useState } from 'react';
import { DataTable } from '@/components/DataTable/DataTable';
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

export function EbayTab({ isActive, endpoint, refreshEndpoint, columns, title, storageKey }: EbayTabProps) {
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
        <button
          onClick={refresh}
          disabled={refreshing}
          className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:bg-gray-400 text-sm"
        >
          {refreshing ? 'Refreshing...' : '🔄 Refresh'}
        </button>
      </div>

      {error && <div className="mb-4 text-red-500 text-sm">{error}</div>}

      {loading ? (
        <p className="p-4 text-gray-500">Loading...</p>
      ) : (
        <DataTable data={allResults} columns={columns} />
      )}
    </div>
  );
}