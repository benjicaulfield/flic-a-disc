import { useEffect, useState } from 'react';
import { apiFetch } from '../../api/client';

interface OOFRecord {
  id: number;
  discogs_id: string;
  artist: string;
  title: string;
  label: string;
  year: number | null;
  wants: number;
  haves: number;
  genres: string[];
  styles: string[];
  suggested_price: string;
  evaluated: boolean;
  wanted: boolean;
}

const PAGE_SIZE = 20;

export default function DiscogsOOF() {
  const [records, setRecords] = useState<OOFRecord[]>([]);
  const [offset, setOffset] = useState(0);
  const [total, setTotal] = useState(0);
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadBatch(offset);
  }, []);

  const loadBatch = async (off: number) => {
    setLoading(true);
    setError(null);
    try {
      const resp = await apiFetch(`/api/discogs/oof?offset=${off}`, { credentials: 'include' });
      if (!resp.ok) throw new Error('Failed to load');
      const data = await resp.json();
      setRecords(data.records || []);
      setTotal(data.total || 0);
      setOffset(off);
      setSelected({});
    } catch (e) {
      setError('Failed to load batch');
    } finally {
      setLoading(false);
    }
  };

  const toggle = (discogsId: string) => {
    setSelected(prev => ({ ...prev, [discogsId]: !prev[discogsId] }));
  };

  const submit = async () => {
    setSaving(true);
    try {
      const labels = records.map(r => ({
        id: r.id,
        label: selected[r.discogs_id] || false,
      }));
      const recordsPayload = records.map(r => ({
        release_id: r.discogs_id,
        artist: r.artist,
        title: r.title,
        label: r.label,
        wants: r.wants,
        haves: r.haves,
        year: r.year,
        genres: r.genres,
        styles: r.styles,
        suggested_price: r.suggested_price,
      }));

      await apiFetch('/api/discogs/catalog/labels', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ labels, records: recordsPayload }),
      });

      loadBatch(offset + PAGE_SIZE);
    } catch (e) {
      setError('Failed to save');
    } finally {
      setSaving(false);
    }
  };

  const progress = total > 0 ? Math.round((offset / total) * 100) : 0;

  if (loading) return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
        <div className="text-gray-600">Loading records...</div>
      </div>
    </div>
  );

  if (error) return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="bg-white rounded-lg shadow p-8 text-center">
        <div className="text-red-600 font-semibold mb-2">Error</div>
        <div className="text-gray-700">{error}</div>
      </div>
    </div>
  );

  if (records.length === 0) return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="bg-white rounded-lg shadow p-12 text-center">
        <div className="text-3xl font-bold text-gray-800 mb-4">All done!</div>
        <div className="text-gray-600">All {total.toLocaleString()} OOF records reviewed.</div>
      </div>
    </div>
  );

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-3xl font-bold text-gray-900">OOF Predictions</h1>
          <div className="text-sm text-gray-600">
            <span className="font-semibold text-blue-600">{offset.toLocaleString()} / {total.toLocaleString()}</span>
            <span className="ml-2 text-gray-400">({progress}%)</span>
          </div>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div className="bg-blue-600 h-2 rounded-full transition-all" style={{ width: `${progress}%` }} />
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full table-fixed divide-y divide-gray-200 text-xs">
            <thead className="bg-gray-50">
              <tr>
                <th className="w-12 px-2 py-2 text-left font-medium text-gray-500 uppercase">Keep</th>
                <th className="w-32 px-2 py-2 text-left font-medium text-gray-500 uppercase">Artist</th>
                <th className="w-32 px-2 py-2 text-left font-medium text-gray-500 uppercase">Title</th>
                <th className="w-12 px-2 py-2 text-left font-medium text-gray-500 uppercase">Year</th>
                <th className="w-28 px-2 py-2 text-left font-medium text-gray-500 uppercase">Label</th>
                <th className="w-14 px-2 py-2 text-center font-medium text-gray-500 uppercase">Want</th>
                <th className="w-14 px-2 py-2 text-center font-medium text-gray-500 uppercase">Have</th>
                <th className="w-20 px-2 py-2 text-left font-medium text-gray-500 uppercase">Genre</th>
                <th className="w-20 px-2 py-2 text-left font-medium text-gray-500 uppercase">Style</th>
                <th className="w-20 px-2 py-2 text-right font-medium text-gray-500 uppercase">Sugg</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {records.map((record, index) => (
                <tr key={record.discogs_id} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-2 py-2">
                    <button
                      className={`w-6 h-6 rounded border flex items-center justify-center text-xs transition-colors ${
                        selected[record.discogs_id]
                          ? 'bg-green-500 border-green-500 text-white'
                          : 'bg-white border-gray-300 hover:border-gray-400'
                      }`}
                      onClick={() => toggle(record.discogs_id)}
                    >
                      {selected[record.discogs_id] && '✓'}
                    </button>
                  </td>
                  <td className="px-2 py-2 font-medium text-gray-900 truncate" title={record.artist}>{record.artist}</td>
                  <td className="px-2 py-2 text-gray-700 truncate" title={record.title}>{record.title}</td>
                  <td className="px-2 py-2 text-gray-700">{record.year}</td>
                  <td className="px-2 py-2 text-gray-700 truncate" title={record.label}>{record.label}</td>
                  <td className="px-2 py-2 text-center">
                    <span className="px-1 py-0.5 rounded text-xs bg-blue-100 text-blue-800">{record.wants}</span>
                  </td>
                  <td className="px-2 py-2 text-center">
                    <span className="px-1 py-0.5 rounded text-xs bg-gray-100 text-gray-800">{record.haves}</span>
                  </td>
                  <td className="px-2 py-2 text-gray-700 truncate" title={record.genres.join(', ')}>
                    {record.genres[0]}{record.genres.length > 1 && <span className="text-gray-400">+{record.genres.length - 1}</span>}
                  </td>
                  <td className="px-2 py-2 text-gray-700 truncate" title={record.styles.join(', ')}>
                    {record.styles[0]}{record.styles.length > 1 && <span className="text-gray-400">+{record.styles.length - 1}</span>}
                  </td>
                  <td className="px-2 py-2 text-gray-500 text-right">
                    {record.suggested_price
                      ? `$${parseFloat(record.suggested_price.replace(/[^0-9.]/g, '')).toFixed(2)}`
                      : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mt-6 flex justify-center">
        <button
          onClick={submit}
          disabled={saving}
          className={`px-8 py-3 rounded-lg text-white font-medium transition-colors ${
            saving ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {saving ? (
            <div className="flex items-center">
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2"></div>
              Saving...
            </div>
          ) : 'Submit & Next'}
        </button>
      </div>
    </div>
  );
}
