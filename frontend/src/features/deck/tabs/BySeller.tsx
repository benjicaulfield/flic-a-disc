import { useState } from 'react';
import { DataTable } from '@/components/DataTable/DataTable';
import { apiFetch } from '../../../api/client';
import type { ColumnDef } from '@tanstack/react-table'
import type { DiscogsListing } from '../../../types/discogs';
import GenericForm from '../../../components/Form';
import { joinList, money } from '../utils';


const STORAGE_KEY = 'discogs_by_seller_results'
type SellerValues = { sellername: string; };

const columns: ColumnDef<DiscogsListing>[] = [
  { accessorKey: 'artist',           header: 'Artist',     size: 200 },
  { accessorKey: 'title',            header: 'Title',      size: 220 },
  { accessorKey: 'format',           header: 'Format',     size: 70,
    cell: ({ getValue }) => joinList(getValue()) },
  { accessorKey: 'label',            header: 'Label',      size: 160 },
  { accessorKey: 'year',             header: 'Year',       size: 60,
    cell: ({ getValue }) => getValue<number | null>() ?? 'N/A' },
  { accessorKey: 'genres',           header: 'Genre',      size: 120,
    cell: ({ getValue }) => joinList(getValue()) },
  { accessorKey: 'styles',           header: 'Style',      size: 120,
    cell: ({ getValue }) => joinList(getValue()) },
  { accessorKey: 'wants',            header: 'Wants',      size: 65 },
  { accessorKey: 'haves',            header: 'Haves',      size: 65 },
  { accessorKey: 'media_condition',  header: 'Cond.',      size: 80 },
  { accessorKey: 'suggested_price',  header: 'Sugg.$',     size: 80,
    cell: ({ getValue }) => money(getValue<number>()) },
  { accessorKey: 'record_price',     header: 'Price (USD)', size: 90,
    cell: ({ getValue }) => money(getValue<string>()) },
  {
    accessorKey: 'score',
    header: 'Score',
    size: 70,
    cell: ({ getValue }) => getValue<number>()?.toFixed(3),
  },
];

function loadCached(): { seller: string; results: DiscogsListing[] } | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function BySeller(_props: { isActive: boolean }) {
  const [seller, setSeller] = useState<string>(() => loadCached()?.seller ?? '');
  const [allResults, setAllResults] = useState<DiscogsListing[]>(
    () => loadCached()?.results ?? [],
  );
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const onSubmit = async (data: SellerValues) => {
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

      if (!response.ok) throw new Error("Request failed");
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
      {seller && <h1 className="text-3xl font-bold mb-6">BY SELLER</h1>}
      <div>
        <GenericForm<SellerValues> 
          fields={[{ name: 'sellername', label: 'Seller', required: true }]}
          onSubmit={onSubmit}
        />
      </div>

      {error && <div className="mb-4 text-red-500 text-sm">{error}</div>}

      {loading ? (
        <p className="p-4 text-gray-500">Loading...</p>
      ) : (
        <DataTable data={allResults} columns={columns} />
      )}

      {allResults.length > 0 && (
        <DataTable data={allResults} columns={columns} />
      )}  
    </div>
  );
}
