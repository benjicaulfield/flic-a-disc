import { useRef, useState } from 'react';
import { DataTable } from '@/components/DataTable/DataTable';
import { apiFetch } from '../../../api/client';
import type { ColumnDef } from '@tanstack/react-table';
import type { NewArrivalRow, NewArrivalsResponse } from '../../../types/discogs';
import { money } from '../utils';

const pct = (v: number | null | undefined) => (v == null ? 'N/A' : `${v.toFixed(1)}%`);

const columns: ColumnDef<NewArrivalRow>[] = [
  { accessorKey: 'artist', header: 'Artist', size: 160 },
  { accessorKey: 'title', header: 'Title', size: 180 },
  { accessorKey: 'seller', header: 'Seller', size: 130 },
  { accessorKey: 'media_condition', header: 'Cond.', size: 90 },
  { accessorKey: 'suggested_price', header: 'Sugg.$', size: 80, cell: ({ getValue }) => money(getValue<number>()) },
  { accessorKey: 'price', header: 'Price (USD)', size: 90, cell: ({ getValue }) => money(getValue<number>()) },
  { accessorKey: 'price_delta_pct', header: 'Δ vs Sugg.', size: 90, cell: ({ getValue }) => pct(getValue<number | null>()) },
  {
    accessorKey: 'listed_date', header: 'Listed', size: 130,
    cell: ({ getValue }) => {
      const v = getValue<string | null>();
      return v ? new Date(v).toLocaleString() : 'N/A';
    },
  },
  {
    id: 'link', header: '', size: 60,
    cell: ({ row }) => (
      <a href={row.original.listing_url} target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">
        view
      </a>
    ),
  },
];

export function NewArrivalsDropzone({ onImported }: { onImported?: () => void }) {
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<NewArrivalsResponse | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const submitFiles = async (files: FileList | File[]) => {
    const validFiles = Array.from(files).filter(f => /\.(csv|json)$/i.test(f.name));
    if (validFiles.length === 0) {
      setError('No .csv or .json files found in the drop.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    validFiles.forEach(f => formData.append('files', f));

    try {
      const response = await apiFetch('api/discogs/wantlist/new-arrivals', {
        method: 'POST',
        credentials: 'include',
        body: formData,
      });
      const json = await response.json();
      if (!response.ok) throw new Error(json.error ?? 'Request failed');
      setResult(json);
      onImported?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process the dropped files.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mb-6">
      <div
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => {
          e.preventDefault();
          setDragging(false);
          if (e.dataTransfer.files.length) submitFiles(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer rounded-lg border-2 border-dashed p-6 text-center text-sm transition ${
          dragging ? 'border-blue-500 bg-blue-50 text-blue-700' : 'border-gray-300 text-gray-500 hover:border-gray-400'
        }`}
      >
        {loading
          ? 'Importing and checking for new arrivals... this can take a few minutes if there are new releases to look up.'
          : 'Drop any number of wantlist JSON or CSV files here, or click to browse'}
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.json"
          multiple
          hidden
          onChange={e => { if (e.target.files?.length) submitFiles(e.target.files); e.target.value = ''; }}
        />
      </div>

      {error && <div className="mt-2 text-sm text-red-500">{error}</div>}

      {result && (
        <div className="mt-4">
          <p className="mb-2 text-sm text-gray-600">
            {result.files_processed} file(s), {result.rows_parsed} rows parsed —{' '}
            <span className="font-semibold">{result.new_arrivals_count} new arrival(s)</span>
          </p>
          {result.new_arrivals_count > 0 && (
            <DataTable data={result.results} columns={columns} />
          )}
        </div>
      )}
    </div>
  );
}
