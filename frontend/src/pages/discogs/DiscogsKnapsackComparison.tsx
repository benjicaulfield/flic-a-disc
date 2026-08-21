import { useState, useEffect } from 'react';
import { apiFetch } from '../../api/client';

interface KnapsackSession {
  id: number;
  seller_name: string;
  budget: number;
  total_cost: number;
  total_score: number;
  selected_count: number;
  created_at: string;
  saved_for_comparison: boolean;
  notes: string;
}

const DiscogsKnapsackComparison = () => {
  const [sessions, setSessions] = useState<KnapsackSession[]>([]);
  const [selectedSessions, setSelectedSessions] = useState<Set<number>>(new Set());
  const [comparisonData, setComparisonData] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSessions();
  }, []);

  const fetchSessions = async () => {
    try {
      const response = await apiFetch('api/discogs/knapsack/sessions?saved=true', {
        credentials: 'include'
      });
      const data = await response.json();
      setSessions(data);
      setLoading(false);
    } catch (err) {
      setError('Failed to load sessions');
      setLoading(false);
    }
  };

  const toggleSession = (sessionId: number) => {
    const newSelected = new Set(selectedSessions);
    if (newSelected.has(sessionId)) {
      newSelected.delete(sessionId);
    } else {
      newSelected.add(sessionId);
    }
    setSelectedSessions(newSelected);
  };

  const loadComparison = async () => {
    if (selectedSessions.size === 0) return;

    try {
      const ids = Array.from(selectedSessions).join(',');
      const response = await apiFetch(`api/discogs/knapsack/sessions/compare?ids=${ids}`, {
        credentials: 'include'
      });
      const data = await response.json();
      setComparisonData(data);
    } catch (err) {
      setError('Failed to load comparison data');
    }
  };

  if (loading) {
    return <div className="container mx-auto p-6">Loading...</div>;
  }

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Knapsack Comparison</h1>

      {error && (
        <div className="mb-4 p-4 bg-red-100 text-red-700 rounded">
          {error}
        </div>
      )}

      {/* Session Selection */}
      <div className="mb-6 bg-white shadow-md rounded p-6">
        <h2 className="text-xl font-bold mb-4">Select Sessions to Compare</h2>

        {sessions.length === 0 ? (
          <p className="text-gray-600">No saved sessions. Save a knapsack result to get started.</p>
        ) : (
          <>
            <div className="space-y-2 mb-4">
              {sessions.map((session) => (
                <label key={session.id} className="flex items-center p-3 border rounded hover:bg-gray-50 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedSessions.has(session.id)}
                    onChange={() => toggleSession(session.id)}
                    className="mr-3 h-4 w-4"
                  />
                  <div className="flex-1">
                    <div className="font-semibold">
                      {session.seller_name} - ${session.total_cost.toFixed(2)} / ${session.budget.toFixed(2)}
                    </div>
                    <div className="text-sm text-gray-600">
                      {session.selected_count} items | Score: {session.total_score.toFixed(2)} | {new Date(session.created_at).toLocaleDateString()}
                    </div>
                    {session.notes && (
                      <div className="text-sm text-gray-500 italic">{session.notes}</div>
                    )}
                  </div>
                </label>
              ))}
            </div>

            <button
              onClick={loadComparison}
              disabled={selectedSessions.size === 0}
              className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50"
            >
              Compare Selected ({selectedSessions.size})
            </button>
          </>
        )}
      </div>

      {/* Comparison Results */}
      {comparisonData.length > 0 && (
        <div className="bg-white shadow-md rounded p-6">
          <h2 className="text-xl font-bold mb-4">Comparison Results</h2>

          {/* Summary Table */}
          <div className="mb-6 overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-100">
                <tr>
                  <th className="px-4 py-2 text-left">Seller</th>
                  <th className="px-4 py-2 text-left">Budget</th>
                  <th className="px-4 py-2 text-left">Total Cost</th>
                  <th className="px-4 py-2 text-left">Items</th>
                  <th className="px-4 py-2 text-left">Total Score</th>
                  <th className="px-4 py-2 text-left">Avg Score/Item</th>
                  <th className="px-4 py-2 text-left">Date</th>
                </tr>
              </thead>
              <tbody>
                {comparisonData.map((session) => (
                  <tr key={session.id} className="border-b">
                    <td className="px-4 py-2 font-semibold">{session.seller_name}</td>
                    <td className="px-4 py-2">${session.budget.toFixed(2)}</td>
                    <td className="px-4 py-2">${session.total_cost.toFixed(2)}</td>
                    <td className="px-4 py-2">{session.selected_count}</td>
                    <td className="px-4 py-2">{session.total_score.toFixed(2)}</td>
                    <td className="px-4 py-2">{(session.total_score / session.selected_count).toFixed(3)}</td>
                    <td className="px-4 py-2">{new Date(session.created_at).toLocaleDateString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Detailed Item Listings */}
          {comparisonData.map((session) => (
            <div key={session.id} className="mb-8">
              <h3 className="text-lg font-bold mb-3 bg-gray-100 p-3 rounded">
                {session.seller_name} ({new Date(session.created_at).toLocaleDateString()})
              </h3>

              <div className="overflow-x-auto">
                <table className="min-w-full text-xs">
                  <thead className="bg-green-50">
                    <tr>
                      <th className="px-2 py-2 text-left">Artist</th>
                      <th className="px-2 py-2 text-left">Title</th>
                      <th className="px-2 py-2 text-left">Label</th>
                      <th className="px-2 py-2 text-left">Year</th>
                      <th className="px-2 py-2 text-left">Genre</th>
                      <th className="px-2 py-2 text-left">Wants</th>
                      <th className="px-2 py-2 text-left">Haves</th>
                      <th className="px-2 py-2 text-left">Price</th>
                      <th className="px-2 py-2 text-left">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {session.selected_items?.map((item: any, i: number) => (
                      <tr key={i} className="border-b hover:bg-gray-50">
                        <td className="px-2 py-2">{item.artist}</td>
                        <td className="px-2 py-2">{item.title}</td>
                        <td className="px-2 py-2">{item.label}</td>
                        <td className="px-2 py-2">{item.year || 'N/A'}</td>
                        <td className="px-2 py-2">{item.genres?.join(', ') || 'N/A'}</td>
                        <td className="px-2 py-2">{item.wants}</td>
                        <td className="px-2 py-2">{item.haves}</td>
                        <td className="px-2 py-2">${item.price.toFixed(2)}</td>
                        <td className="px-2 py-2">{item.score.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default DiscogsKnapsackComparison;
