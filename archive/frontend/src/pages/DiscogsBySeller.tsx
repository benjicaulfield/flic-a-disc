import { useState } from 'react';
import { DataTable } from '@/components/DataTable/DataTable'
import { apiFetch } from '../../../../frontend/src/api/client';
import type { ColumnDef } from '@tanstack/react-table'
import type { DiscogsListing } from '../../../../frontend/src/types/discogs'
import GenericForm from '../../../../frontend/src/components/Form'

type SellerValues = {
  sellername: string;
};

const columns: ColumnDef<DiscogsListing>[] = [
  { accessorKey: 'artist', header: 'Artist', size: 200 },
  { accessorKey: 'title', header: 'Title', size: 220 },
  {
    accessorKey: 'format',
    header: 'Format',
    size: 70,
    cell: ({ getValue }) => (getValue<string[]>() ?? []).join(', ') || 'N/A',
  },
  { accessorKey: 'label', header: 'Label', size: 160 },
  {
    accessorKey: 'year',
    header: 'Year',
    size: 60,
    cell: ({ getValue }) => getValue<number | null>() ?? 'N/A',
  },
  {
    accessorKey: 'genres',
    header: 'Genre',
    size: 120,
    cell: ({ getValue }) => (getValue<string[]>() ?? []).join(', ') || 'N/A',
  },
  { accessorKey: 'media_condition', header: 'Cond.', size: 80 },
  { accessorKey: 'wants', header: 'Wants', size: 65 },
  { accessorKey: 'haves', header: 'Haves', size: 65 },
  {
    accessorKey: 'suggested_price',
    header: 'Sugg.$',
    size: 80,
    cell: ({ getValue }) => {
      const v = getValue<number>();
      return v ? `$${v.toFixed(2)}` : 'N/A';
    },
  },
  {
    accessorKey: 'price',
    header: 'Price (USD)',
    size: 90,
    cell: ({ getValue }) => `$${getValue<number>()?.toFixed(2)}`,
  },
  {
    accessorKey: 'score',
    header: 'Score',
    size: 70,
    cell: ({ getValue }) => getValue<number>()?.toFixed(3),
  },
];

const STORAGE_KEY = 'discogs_by_seller_results';

function loadCached(): { seller: string; results: DiscogsBySellerListing[] } | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export default function DiscogsBySeller() {
  const cached = loadCached();
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [seller, setSeller] = useState<string>(cached?.seller ?? '');
  const [allResults, setAllResults] = useState<DiscogsBySellerListing[]>(cached?.results ?? []);

  const onSubmit = async (data:SellerValues) => {
    setError(null);
    setLoading(true);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 25 * 60 * 1000);

    try {
      const response = await apiFetch("api/discogs/by-seller", {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        signal: controller.signal,
        body: JSON.stringify({ seller: data.sellername }),
      });
      clearTimeout(timeoutId);

      const json = await response.json();
      const results = json.results ?? [];
      const sellerName = json.seller ?? '';
      setAllResults(results);
      setSeller(sellerName);
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ seller: sellerName, results }));
    } catch (err) {
      clearTimeout(timeoutId);
      if (err instanceof Error && err.name === 'AbortError') {
        setError("Request timed out after 25 minutes.");
      } else {
        setError("Request failed.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full px-6 py-6">
      <h1 className="text-3xl font-bold mb-6">BY SELLER</h1>
      <div>
        <GenericForm<SellerValues> 
          fields={[{ name: 'sellername', label: 'Seller', required: true }]}
          onSubmit={onSubmit}
        />
      </div>

      {error && <div className="mb-4 text-red-500 text-sm">{error}</div>}

      {allResults.length > 0 && (
        <DataTable data={allResults} columns={columns} />
      )}  
    </div>
  );
};

