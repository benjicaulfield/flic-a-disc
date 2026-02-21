import { useState, type FormEvent } from 'react';
import { apiFetch } from '../api/client';

const DiscogsKnapsack = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [seller, setSeller] = useState<string>("")
  const [budget, setBudget] = useState<string>("");
  const [results, setResults] = useState<any>(null);

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
      const timeoutId = setTimeout(() => controller.abort(), 15 * 60 * 1000);

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
      console.log(data)
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">KNAPSACK PROBLEM</h1>
      
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
          <h2 className="text-2xl font-bold mb-4">
            Results for {results.seller} - ${results.total_cost?.toFixed(2) ?? '0.00'} / ${budget}
          </h2>

          <div className="bg-white shadow-md rounded p-6">
            {/* Summary Stats */}
            <div className="grid grid-cols-3 gap-4 mb-6 bg-gray-50 p-4 rounded">
              <div>
                <p className="text-sm text-gray-600">Items Selected</p>
                <p className="text-2xl font-bold">{results.total_selected}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Total Cost</p>
                <p className="text-2xl font-bold">${results.total_cost.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Total Score</p>
                <p className="text-2xl font-bold">{results.total_score.toFixed(2)}</p>
              </div>
            </div>

            {/* Knapsack Items */}
            <div className="mb-6">
              <h4 className="font-bold text-lg mb-2">🎯 Selected Items</h4>
              <div className="overflow-x-auto">
                <table className="min-w-full">
                  <thead className="bg-green-100">
                    <tr>
                      <th className="px-4 py-2 text-left">Artist</th>
                      <th className="px-4 py-2 text-left">Title</th>
                      <th className="px-4 py-2 text-left">Condition</th>
                      <th className="px-4 py-2 text-left">Wants</th>
                      <th className="px-4 py-2 text-left">Haves</th>
                      <th className="px-4 py-2 text-left">Sugg. Price</th>
                      <th className="px-4 py-2 text-left">Price</th>
                      <th className="px-4 py-2 text-left">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.knapsack.map((item: any, i: number) => (
                      <tr key={i} className="border-b hover:bg-gray-50">
                        <td className="px-4 py-2">{item.artist}</td>
                        <td className="px-4 py-2">{item.title}</td>
                        <td className="px-4 py-2">{item.media_condition}</td>
                        <td className="px-4 py-2">{item.wants}</td>
                        <td className="px-4 py-2">{item.haves}</td>
                        <td className="px-4 py-2">${item.suggested_price?.toFixed(2) || 'N/A'}</td>
                        <td className="px-4 py-2">${item.price.toFixed(2)} {item.currency}</td>
                        <td className="px-4 py-2">{item.score.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Contenders */}
            {results.contenders.length > 0 && (
              <div>
                <h4 className="font-bold text-lg mb-2">📋 Top Contenders (Not Selected)</h4>
                <div className="overflow-x-auto">
                  <table className="min-w-full">
                    <thead className="bg-gray-100">
                      <tr>
                        <th className="px-4 py-2 text-left">Artist</th>
                        <th className="px-4 py-2 text-left">Title</th>
                        <th className="px-4 py-2 text-left">Condition</th>
                        <th className="px-4 py-2 text-left">Wants</th>
                        <th className="px-4 py-2 text-left">Haves</th>
                        <th className="px-4 py-2 text-left">Sugg. Price</th>
                        <th className="px-4 py-2 text-left">Price</th>
                        <th className="px-4 py-2 text-left">Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.contenders.map((item: any, i: number) => (
                        <tr key={i} className="border-b hover:bg-gray-50">
                          <td className="px-4 py-2">{item.artist}</td>
                          <td className="px-4 py-2">{item.title}</td>
                          <td className="px-4 py-2">{item.media_condition}</td>
                          <td className="px-4 py-2">{item.wants}</td>
                          <td className="px-4 py-2">{item.haves}</td>
                          <td className="px-4 py-2">${item.suggested_price?.toFixed(2) || 'N/A'}</td>
                          <td className="px-4 py-2">${item.price.toFixed(2)} {item.currency}</td>
                          <td className="px-4 py-2">{item.score.toFixed(3)}</td>
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

