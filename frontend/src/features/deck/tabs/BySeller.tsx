import { useState } from 'react';
import { DataTable } from '@/components/DataTable/DataTable';
import { apiFetch } from '../../../api/client';
import type { ColumnDef } from '@tanstack/react-table';
import type { DiscogsListing } from '../../../types/discogs';
import GenericForm from '../../../components/Form';
import { joinList, money } from '../utils';

const STORAGE_KEY = 'discogs_by_seller_results'
type SellerValues = { sellername: string; };
type Annotation = { keeper?: boolean; wantlist?: boolean };

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
  const [annotations, setAnnotations] = useState<Record<number, Annotation>>({});
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [submitMessage, setSubmitMessage] = useState<string | null>(null);

  const toggleAnnotation = (row: DiscogsListing, field: keyof Annotation) => {
    setAnnotations(prev => {
      const current = prev[row.listing_id] ?? {};
      const dbDefault = field === 'keeper' ? (row.wanted ?? false) : (row.wantlist ?? false);
      const displayed = current[field] ?? dbDefault;
      return { ...prev, [row.listing_id]: { ...current, [field]: !displayed } };
    });
  };

  const renderAnnotationCell = (row: DiscogsListing, field: keyof Annotation) => {
    const override = annotations[row.listing_id]?.[field];
    const dbValue = field === 'keeper' ? row.wanted : row.wantlist;
    const dbEvaluated = field === 'keeper' ? row.evaluated : row.wantlist_evaluated;

    if (override === undefined && dbEvaluated && !dbValue) {
      return (
        <button
          type="button"
          onClick={() => toggleAnnotation(row, field)}
          title="Evaluated — not a match. Click to change."
          className="h-4 w-4 flex items-center justify-center text-red-400 hover:text-red-500 text-sm leading-none"
        >
          ✕
        </button>
      );
    }

    return (
      <input
        type="checkbox"
        className="h-4 w-4"
        checked={override ?? dbValue ?? false}
        onChange={() => toggleAnnotation(row, field)}
      />
    );
  };

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
    { accessorKey: 'price',            header: 'Price (USD)', size: 90,
      cell: ({ getValue }) => money(getValue<number>()) },
    {
      id: 'keeper',
      header: 'Keeper',
      size: 30,
      cell: ({ row }) => renderAnnotationCell(row.
        
        original, 'keeper'),
    },
    {
      id: 'wantlist',
      header: 'Wantlist',
      size: 30,
      cell: ({ row }) => renderAnnotationCell(row.original, 'wantlist'),
    },
    {
      accessorKey: 'score',
      header: 'Score',
      size: 70,
      cell: ({ getValue }) => getValue<number>()?.toFixed(3),
    },
  ];

  const onSubmit = async (data: SellerValues) => {
    setError(null);
    setSubmitMessage(null);
    setAnnotations({});
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
        try {
          const savedResponse = await apiFetch(
            `api/discogs/by-seller/saved?seller=${encodeURIComponent(data.sellername)}`,
          );
          const savedJson = await savedResponse.json();
          const savedResults = savedJson.results ?? [];
          if (savedResponse.ok && savedResults.length > 0) {
            const sellerName = savedJson.seller ?? data.sellername;
            setAllResults(savedResults);
            setSeller(sellerName);
            sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ seller: sellerName, results: savedResults }));
            setError("Request timed out after 25 minutes. Showing partial results saved during the scrape.");
          } else {
            setError("Request timed out after 25 minutes.");
          }
        } catch {
          setError("Request timed out after 25 minutes.");
        }
      } else {
        setError("Request failed.");
      }
    } finally {
      setLoading(false);
    }
  };

  const submitAnnotations = async () => {
    if (allResults.length === 0) return;

    setSubmitting(true);
    setSubmitMessage(null);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 25 * 60 * 1000);

    try {
      const response = await apiFetch("api/discogs/annotate", {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        signal: controller.signal,
        body: JSON.stringify({
          // Every row in this batch gets a keeper decision — explicit True/False if you
          // toggled it, implicit False otherwise. An unclicked row means "reviewed, not a
          // match," not "skip this one" — it must count as a negative example.
          annotations: allResults.map(r => {
            const a = annotations[r.listing_id] ?? {};
            return {
              discogs_id: r.discogs_id,
              keeper: a.keeper ?? false,
              ...(a.wantlist !== undefined ? { wantlist: a.wantlist } : {}),
              artist: r.artist,
              title: r.title,
              label: r.label,
              catno: r.catno,
              wants: r.wants,
              haves: r.haves,
              genres: r.genres,
              styles: r.styles,
              year: r.year,
              suggested_price: r.suggested_price,
              probability: r.probability,
              uncertainty: r.uncertainty,
            };
          }),
        }),
      });

      clearTimeout(timeoutId);
      const json = await response.json();
      if (!response.ok) throw new Error("Request failed");

      setAnnotations({});
      const errors: string[] = json.errors ?? [];
      if (errors.length) {
        console.error('discogs/annotate errors:', errors);
      }
      setSubmitMessage(
        `Saved ${json.keepers ?? 0} keeper(s), added ${json.wantlisted ?? 0} to wantlist.` +
        (errors.length ? ` ${errors.length} error(s): ${errors.join('; ')}` : '') +
        (json.training?.model_updated
          ? ` Model retrained: ${(json.training.accuracy * 100).toFixed(1)}% batch accuracy, threshold now ${json.training.updated_threshold.toFixed(2)}.`
          : ''),
      );
    } catch (err) {
      clearTimeout(timeoutId);
      console.error('discogs/annotate request failed:', err);
      if (err instanceof Error && err.name === 'AbortError') {
        setSubmitMessage("Request timed out after 25 minutes.");
      } else {
        setSubmitMessage("Failed to submit annotations.");
      }
    } finally {
      setSubmitting(false);
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
      {submitMessage && <div className="mb-4 text-sm text-gray-700">{submitMessage}</div>}

      {loading ? (
        <p className="p-4 text-gray-500">Loading...</p>
      ) : (
        allResults.length > 0 && (
          <>
            <div className="mb-4">
              <button
                onClick={submitAnnotations}
                disabled={submitting || Object.keys(annotations).length === 0}
                className="px-4 py-2 text-sm rounded border bg-blue-600 text-white border-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
              >
                {submitting ? 'Submitting...' : 'Submit Annotations'}
              </button>
            </div>
            <DataTable data={allResults} columns={columns} />
          </>
        )
      )}
    </div>
  );
}
