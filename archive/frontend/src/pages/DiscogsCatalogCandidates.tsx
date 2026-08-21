import { useEffect, useState, Fragment } from 'react';
import { apiFetch } from '../../../../frontend/src/api/client';

interface CandidateRecord {
  id: number;
  discogs_id: string;
  artist: string;
  title: string;
  label: string;
  catno: string;
  year: number | null;
  genres: string[];
  styles: string[];
  wants: number;
  haves: number;
  suggested_price: string;
}

const DiscogsCatalogCandidates = () => {
  const [records, setRecords] = useState<CandidateRecord[]>([]);
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [lastClickedIndex, setLastClickedIndex] = useState<number | null>(null);
  const [total, setTotal] = useState(0);
  const [annotated, setAnnotated] = useState(0);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadBatch();
  }, []);

  const loadBatch = async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await apiFetch('/api/discogs/catalog-candidates', { credentials: 'include' });
      if (!resp.ok) throw new Error('Failed to load');
      const data = await resp.json();
      setRecords(data.records);
      setTotal(data.total);
      setAnnotated(data.annotated);
      setSelected({});
      setLastClickedIndex(null);
    } catch {
      setError('Failed to load candidates');
    } finally {
      setLoading(false);
    }
  };

  const toggle = (discogsId: string, index: number, e: React.MouseEvent) => {
    if (e.shiftKey && lastClickedIndex !== null) {
      const start = Math.min(lastClickedIndex, index);
      const end = Math.max(lastClickedIndex, index);
      const shouldSelect = !selected[discogsId];
      setSelected(prev => {
        const next = { ...prev };
        for (let i = start; i <= end; i++) {
          if (records[i]) next[records[i].discogs_id] = shouldSelect;
        }
        return next;
      });
    } else {
      setSelected(prev => ({ ...prev, [discogsId]: !prev[discogsId] }));
    }
    setLastClickedIndex(index);
  };

  const saveBatch = async () => {
    setSaving(true);
    try {
      const labels = records.map(r => ({
        discogs_id: r.discogs_id,
        wanted: selected[r.discogs_id] || false,
      }));
      const resp = await apiFetch('/api/discogs/catalog-candidates', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ labels }),
      });
      if (!resp.ok) throw new Error('Save failed');
      await loadBatch();
    } catch {
      setError('Failed to save');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-white rounded-lg shadow-md p-8 text-center">
          <div className="text-red-600 font-semibold mb-2">Error</div>
          <div className="text-gray-700">{error}</div>
        </div>
      </div>
    );
  }

  if (records.length === 0) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-white rounded-lg shadow-md p-12 text-center">
          <div className="text-3xl font-bold text-gray-800 mb-4">Done!</div>
          <div className="text-gray-600">All {total} six-vote candidates annotated.</div>
        </div>
      </div>
    );
  }

  const selectedCount = Object.values(selected).filter(Boolean).length;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4">
          <h1 className="text-3xl font-bold text-gray-900">Catalog Candidates</h1>
          <div className="mt-2 sm:mt-0 text-sm text-gray-600">
            <span className="font-semibold text-blue-600">{annotated}</span> / {total} annotated
            {selectedCount > 0 && (
              <span className="ml-4 text-green-600 font-semibold">{selectedCount} selected</span>
            )}
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full table-fixed divide-y divide-gray-200 text-xs">
            <thead className="bg-gray-50">
              <tr>
                <th className="w-10 px-2 py-2 text-left font-medium text-gray-500 uppercase">✓</th>
                <th className="w-32 px-2 py-2 text-left font-medium text-gray-500 uppercase">Artist</th>
                <th className="w-32 px-2 py-2 text-left font-medium text-gray-500 uppercase">Title</th>
                <th className="w-12 px-2 py-2 text-left font-medium text-gray-500 uppercase">Year</th>
                <th className="w-28 px-2 py-2 text-left font-medium text-gray-500 uppercase">Label</th>
                <th className="w-20 px-2 py-2 text-left font-medium text-gray-500 uppercase">Cat#</th>
                <th className="w-14 px-2 py-2 text-center font-medium text-gray-500 uppercase">Want</th>
                <th className="w-14 px-2 py-2 text-center font-medium text-gray-500 uppercase">Have</th>
                <th className="w-20 px-2 py-2 text-left font-medium text-gray-500 uppercase">Genre</th>
                <th className="w-20 px-2 py-2 text-left font-medium text-gray-500 uppercase">Style</th>
                <th className="w-20 px-2 py-2 text-right font-medium text-gray-500 uppercase">Sugg</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {records.map((record, index) => (
                <Fragment key={record.discogs_id}>
                  <tr className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-2 py-2">
                      <button
                        className={`w-6 h-6 rounded border flex items-center justify-center text-xs transition-colors ${
                          selected[record.discogs_id]
                            ? 'bg-green-500 border-green-500 text-white hover:bg-green-600'
                            : 'bg-white border-gray-300 hover:border-gray-400 hover:bg-gray-50'
                        }`}
                        onClick={e => toggle(record.discogs_id, index, e)}
                      >
                        {selected[record.discogs_id] && '✓'}
                      </button>
                    </td>
                    <td className="px-2 py-2 font-medium text-gray-900 truncate" title={record.artist}>
                      {record.artist}
                    </td>
                    <td className="px-2 py-2 text-gray-700 truncate" title={record.title}>
                      {record.title}
                    </td>
                    <td className="px-2 py-2 text-gray-700">{record.year}</td>
                    <td className="px-2 py-2 text-gray-700 truncate" title={record.label}>
                      {record.label}
                    </td>
                    <td className="px-2 py-2 text-gray-500 truncate" title={record.catno}>
                      {record.catno}
                    </td>
                    <td className="px-2 py-2 text-center">
                      <span className="px-1 py-0.5 rounded text-xs bg-blue-100 text-blue-800">
                        {record.wants}
                      </span>
                    </td>
                    <td className="px-2 py-2 text-center">
                      <span className="px-1 py-0.5 rounded text-xs bg-gray-100 text-gray-800">
                        {record.haves}
                      </span>
                    </td>
                    <td className="px-2 py-2 text-gray-700 truncate" title={record.genres.join(', ')}>
                      {record.genres[0]}
                      {record.genres.length > 1 && <span className="text-gray-400">+{record.genres.length - 1}</span>}
                    </td>
                    <td className="px-2 py-2 text-gray-700 truncate" title={record.styles.join(', ')}>
                      {record.styles[0]}
                      {record.styles.length > 1 && <span className="text-gray-400">+{record.styles.length - 1}</span>}
                    </td>
                    <td className="px-2 py-2 text-gray-500 text-right">
                      {record.suggested_price
                        ? `$${parseFloat(record.suggested_price.replace(/[^0-9.]/g, '')).toFixed(2)}`
                        : '—'}
                    </td>
                  </tr>
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mt-6 flex justify-center">
        <button
          className={`px-8 py-3 rounded-lg text-white font-medium transition-colors ${
            saving ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800'
          }`}
          onClick={saveBatch}
          disabled={saving}
        >
          {saving ? (
            <div className="flex items-center">
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2" />
              Saving...
            </div>
          ) : (
            `Submit (${selectedCount} keepers)`
          )}
        </button>
      </div>
    </div>
  );
};

export default DiscogsCatalogCandidates;
