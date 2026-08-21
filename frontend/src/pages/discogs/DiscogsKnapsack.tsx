import { useState, type FormEvent } from 'react';
import { Link } from 'react-router-dom';
import { apiFetch } from '../../api/client';

const DiscogsKnapsack = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [seller, setSeller] = useState<string>("")
  const [budget, setBudget] = useState<string>("");
  const [results, setResults] = useState<any>(null);
  const [currentSessionId, setCurrentSessionId] = useState<number | null>(null);
  const [savedForComparison, setSavedForComparison] = useState<boolean>(false);

  const handleSaveForComparison = async () => {
    if (!currentSessionId) return;

    try {
      await apiFetch(`api/discogs/knapsack/sessions/${currentSessionId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ saved_for_comparison: true })
      });
      setSavedForComparison(true);
    } catch (err) {}
  };

  const handleFormSubmit = async (e: FormEvent) => {
    e.preventDefault();

    if (!budget || parseFloat(budget) <= 0) {
      setError('Please enter a valid budget');
      return;
    }
    if (!seller) {
      setError('Please enter seller');
    }
    
    setError(null);
    setLoading(true);

    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 25 * 60 * 1000); // 25 minutes

      const response = await apiFetch("api/discogs/knapsack", {
        method: 'POST',
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ 
          seller: seller,
          budget: parseFloat(budget) })
      });
      clearTimeout(timeoutId);
      const data = await response.json()
      setResults(data);
      setCurrentSessionId(data.session_id);
      setSavedForComparison(false);
    } catch (err) {
      if (err instanceof Error) {
        if (err.name === 'AbortError') {
          setError('Request timed out (>25 minutes). The seller may have too many listings. Try a different seller or increase the timeout.');
        } else {
          setError(err.message);
        }
      } else {
        setError('Something went wrong');
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container mx-auto p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">KNAPSACK PROBLEM</h1>
        <Link
          to="/discogs/knapsack/compare"
          className="bg-purple-500 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded"
        >
          Compare Saved Sessions
        </Link>
      </div>

      <form onSubmit={handleFormSubmit} className="max-w-md bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
        <div className="mb-6">
          <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="budget">
            Budget ($)
          </label>
          <input
            id="budget"
            type="number"
            step="0.01"
            min="0"
            value={budget}
            onChange={(e) => setBudget(e.target.value)}
            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
            placeholder="Enter your budget"
            disabled={loading}
          />
        </div>
        <div className="mb-6">
          <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="budget">
            Seller
          </label>
          <input
            id="seller"
            type="string"
            value={seller}
            onChange={(e) => setSeller(e.target.value)}
            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
            placeholder="Seller"
            disabled={loading}
          />
        </div>
        

        {error && (
          <div className="mb-4 text-red-500 text-sm">
            {error}
          </div>
        )}

        <div className="flex items-center justify-between">
          <button
            type="submit"
            disabled={loading}
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline disabled:opacity-50"
          >
            {loading ? 'hold on....' : 'lets do this....'}
          </button>
        </div>
      </form>

      {/* Results section */}
      {results && (
        <div className="mt-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-bold">
              Results for {results.seller} - ${results.total_cost?.toFixed(2) ?? '0.00'} / ${budget}
            </h2>
            <button
              onClick={handleSaveForComparison}
              disabled={savedForComparison}
              className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {savedForComparison ? '✓ Saved for comparison' : 'Save for comparison'}
            </button>
          </div>

          <div className="bg-white shadow-md rounded p-6">
            {/* Summary Stats */}
            <div className="grid grid-cols-3 gap-4 mb-6 bg-gray-50 p-4 rounded">
              <div>
                <p className="text-sm text-gray-600">Items Selected</p>
                <p className="text-2xl font-bold">{results.total_selected ?? 0}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Total Cost</p>
                <p className="text-2xl font-bold">${results.total_cost?.toFixed(2) ?? '0.00'}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Total Score</p>
                <p className="text-2xl font-bold">{results.total_score?.toFixed(2) ?? '0.00'}</p>
              </div>
            </div>

            {/* Knapsack Items */}
            <div className="mb-6">
              <h4 className="font-bold text-lg mb-2">🎯 Selected Items</h4>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead className="bg-green-100">
                    <tr>
                      <th className="px-2 py-2 text-left">Artist</th>
                      <th className="px-2 py-2 text-left">Title</th>
                      <th className="px-2 py-2 text-left">Label</th>
                      <th className="px-2 py-2 text-left">Year</th>
                      <th className="px-2 py-2 text-left">Genre</th>
                      <th className="px-2 py-2 text-left">Style</th>
                      <th className="px-2 py-2 text-left">Cond.</th>
                      <th className="px-2 py-2 text-left">Wants</th>
                      <th className="px-2 py-2 text-left">Haves</th>
                      <th className="px-2 py-2 text-left">Sugg.$</th>
                      <th className="px-2 py-2 text-left">Price</th>
                      <th className="px-2 py-2 text-left">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.knapsack?.map((item: any, i: number) => (
                      <tr key={i} className="border-b hover:bg-gray-50">
                        <td className="px-2 py-2">{item.artist}</td>
                        <td className="px-2 py-2">{item.title}</td>
                        <td className="px-2 py-2">{item.label}</td>
                        <td className="px-2 py-2">{item.year || 'N/A'}</td>
                        <td className="px-2 py-2">{item.genres?.join(', ') || 'N/A'}</td>
                        <td className="px-2 py-2">{item.styles?.join(', ') || 'N/A'}</td>
                        <td className="px-2 py-2">{item.media_condition}</td>
                        <td className="px-2 py-2">{item.wants}</td>
                        <td className="px-2 py-2">{item.haves}</td>
                        <td className="px-2 py-2">${item.suggested_price?.toFixed(2) || 'N/A'}</td>
                        <td className="px-2 py-2">${item.price.toFixed(2)}</td>
                        <td className="px-2 py-2">{item.score.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Contenders */}
            {results.contenders?.length > 0 && (
              <div>
                <h4 className="font-bold text-lg mb-2">📋 Top Contenders (Not Selected)</h4>
                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead className="bg-gray-100">
                      <tr>
                        <th className="px-2 py-2 text-left">Artist</th>
                        <th className="px-2 py-2 text-left">Title</th>
                        <th className="px-2 py-2 text-left">Label</th>
                        <th className="px-2 py-2 text-left">Year</th>
                        <th className="px-2 py-2 text-left">Genre</th>
                        <th className="px-2 py-2 text-left">Style</th>
                        <th className="px-2 py-2 text-left">Cond.</th>
                        <th className="px-2 py-2 text-left">Wants</th>
                        <th className="px-2 py-2 text-left">Haves</th>
                        <th className="px-2 py-2 text-left">Sugg.$</th>
                        <th className="px-2 py-2 text-left">Price</th>
                        <th className="px-2 py-2 text-left">Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.contenders?.map((item: any, i: number) => (
                        <tr key={i} className="border-b hover:bg-gray-50">
                          <td className="px-2 py-2">{item.artist}</td>
                          <td className="px-2 py-2">{item.title}</td>
                          <td className="px-2 py-2">{item.label}</td>
                          <td className="px-2 py-2">{item.year || 'N/A'}</td>
                          <td className="px-2 py-2">{item.genres?.join(', ') || 'N/A'}</td>
                          <td className="px-2 py-2">{item.styles?.join(', ') || 'N/A'}</td>
                          <td className="px-2 py-2">{item.media_condition}</td>
                          <td className="px-2 py-2">{item.wants}</td>
                          <td className="px-2 py-2">{item.haves}</td>
                          <td className="px-2 py-2">${item.suggested_price?.toFixed(2) || 'N/A'}</td>
                          <td className="px-2 py-2">${item.price.toFixed(2)}</td>
                          <td className="px-2 py-2">{item.score.toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};




export default DiscogsKnapsack;

