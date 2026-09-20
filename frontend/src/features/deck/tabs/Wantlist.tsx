import { useCallback, useEffect, useState } from 'react';
import { DataTable } from '@/components/DataTable/DataTable';
import { apiFetch } from '../../../api/client';
import type { ColumnDef } from '@tanstack/react-table';
import type { WantlistListing, WantlistSellerSummary } from '../../../types/discogs';
import { money } from '../utils';
import { NewArrivalsDropzone } from './NewArrivalsDropzone';

const pct = (v: number | null | undefined) => (v == null ? 'N/A' : `${v.toFixed(1)}%`);

export function Wantlist(_props: { isActive: boolean }) {
  const [listings, setListings] = useState<WantlistListing[]>([]);
  const [sellers, setSellers] = useState<WantlistSellerSummary[]>([]);
  const [selectedSeller, setSelectedSeller] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(() => {
    setLoading(true);
    setError(null);

    return Promise.all([
      apiFetch('api/discogs/wantlist/scored').then(r => r.json()),
      apiFetch('api/discogs/wantlist/sellers').then(r => r.json()),
    ])
      .then(([scoredJson, sellersJson]) => {
        setListings(scoredJson.results ?? []);
        setSellers(sellersJson.results ?? []);
      })
      .catch(() => {
        setError('Failed to load wantlist data.');
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const visibleListings = selectedSeller
    ? listings.filter(l => l.seller === selectedSeller)
    : listings;

  const columns: ColumnDef<WantlistListing>[] = [
    { accessorKey: 'artist', header: 'Artist', size: 180 },
    { accessorKey: 'title', header: 'Title', size: 200 },
    { accessorKey: 'seller', header: 'Seller', size: 140 },
    { accessorKey: 'media_condition', header: 'Cond.', size: 90 },
    {
      accessorKey: 'suggested_price', header: 'Sugg.$', size: 80,
      cell: ({ getValue }) => money(getValue<number>()),
    },
    {
      accessorKey: 'price', header: 'Price (USD)', size: 90,
      cell: ({ getValue }) => money(getValue<number>()),
    },
    {
      accessorKey: 'price_delta_pct', header: 'Δ vs Sugg.', size: 90,
      cell: ({ getValue }) => pct(getValue<number | null>()),
    },
    {
      accessorKey: 'embedding_score', header: 'Embed.', size: 80,
      cell: ({ getValue }) => getValue<number>()?.toFixed(3) ?? 'N/A',
    },
    {
      accessorKey: 'score', header: 'Score', size: 80,
      cell: ({ getValue }) => getValue<number>()?.toFixed(3) ?? 'N/A',
    },
    {
      id: 'link', header: '', size: 60,
      cell: ({ row }) => (
        <a
          href={`https://www.discogs.com/shop/item/${row.original.listing_id}`}
          target="_blank"
          rel="noreferrer"
          className="text-blue-600 hover:underline"
        >
          view
        </a>
      ),
    },
  ];

  return (
    <div className="w-full px-6 py-6 flex gap-6">
      <aside className="w-64 shrink-0">
        <h2 className="text-lg font-semibold mb-2">Sellers</h2>
        {selectedSeller && (
          <button
            onClick={() => setSelectedSeller(null)}
            className="mb-2 text-xs text-blue-600 hover:underline"
          >
            Clear filter ({selectedSeller})
          </button>
        )}
        <div className="border border-gray-200 rounded-lg shadow-sm max-h-[70vh] overflow-y-auto">
          <table className="w-full text-xs">
            <thead className="bg-gray-50 sticky top-0">
              <tr>
                <th className="text-left px-2 py-1 font-semibold text-gray-500">Seller</th>
                <th className="text-right px-2 py-1 font-semibold text-gray-500">#</th>
                <th className="text-right px-2 py-1 font-semibold text-gray-500">Avg Δ%</th>
              </tr>
            </thead>
            <tbody>
              {sellers.map(s => (
                <tr
                  key={s.seller}
                  onClick={() => setSelectedSeller(s.seller === selectedSeller ? null : s.seller)}
                  className={`cursor-pointer hover:bg-gray-100 border-t border-gray-100 ${
                    s.seller === selectedSeller ? 'bg-blue-50' : ''
                  }`}
                >
                  <td className="px-2 py-1 truncate max-w-[120px]" title={s.seller}>{s.seller}</td>
                  <td className="px-2 py-1 text-right">{s.listing_count}</td>
                  <td className="px-2 py-1 text-right">{pct(s.avg_price_delta_pct)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </aside>

      <div className="flex-1 min-w-0">
        <h1 className="text-3xl font-bold mb-6">WANTLIST</h1>
        <NewArrivalsDropzone onImported={loadData} />
        {error && <div className="mb-4 text-red-500 text-sm">{error}</div>}
        {loading ? (
          <p className="p-4 text-gray-500">Loading...</p>
        ) : (
          <DataTable data={visibleListings} columns={columns} />
        )}
      </div>
    </div>
  );
}
