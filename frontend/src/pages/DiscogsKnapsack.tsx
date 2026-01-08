import { useState, type FormEvent } from 'react';
import { apiFetch } from '../api/client';

const DiscogsKnapsack = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [budget, setBudget] = useState<string>("");
  const [results, setResults] = useState<any>(null);

  const handleFormSubmit = async (e: FormEvent) => {
    e.preventDefault();

    if (!budget || parseFloat(budget) <= 0) {
      setError('Please enter a valid budget');
      return;
    }
    
    setError(null);
    setLoading(true);

    try {
      const response = await apiFetch("api/discogs/knapsack", {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ budget: parseFloat(budget) })
      });
      console.log("Response:", response);  // ← Add this
      setResults(response);
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
            Results for ${results.budget} budget
          </h2>

          {results.knapsacks.map((sellerData: any, idx: number) => (
            <div key={idx} className="mb-8 bg-white shadow-md rounded p-6">
              <h3 className="text-xl font-bold mb-4">{sellerData.seller}</h3>
              
              {/* Summary Stats */}
              <div className="grid grid-cols-3 gap-4 mb-6 bg-gray-50 p-4 rounded">
                <div>
                  <p className="text-sm text-gray-600">Items Selected</p>
                  <p className="text-2xl font-bold">{sellerData.total_selected}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Total Cost</p>
                  <p className="text-2xl font-bold">${sellerData.total_cost.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Total Score</p>
                  <p className="text-2xl font-bold">{sellerData.total_score.toFixed(2)}</p>
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
                        <th className="px-4 py-2 text-left">Price</th>
                        <th className="px-4 py-2 text-left">Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sellerData.knapsack.map((item: any, i: number) => (
                        <tr key={i} className="border-b hover:bg-gray-50">
                          <td className="px-4 py-2">{item.artist}</td>
                          <td className="px-4 py-2">{item.title}</td>
                          <td className="px-4 py-2">{item.media_condition}</td>
                          <td className="px-4 py-2">${item.price.toFixed(2)} {item.currency}</td>
                          <td className="px-4 py-2">{item.score.toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Contenders */}
              {sellerData.contenders.length > 0 && (
                <div>
                  <h4 className="font-bold text-lg mb-2">📋 Top Contenders (Not Selected)</h4>
                  <div className="overflow-x-auto">
                    <table className="min-w-full">
                      <thead className="bg-gray-100">
                        <tr>
                          <th className="px-4 py-2 text-left">Artist</th>
                          <th className="px-4 py-2 text-left">Title</th>
                          <th className="px-4 py-2 text-left">Condition</th>
                          <th className="px-4 py-2 text-left">Price</th>
                          <th className="px-4 py-2 text-left">Score</th>
                        </tr>
                      </thead>
                      <tbody>
                        {sellerData.contenders.map((item: any, i: number) => (
                          <tr key={i} className="border-b hover:bg-gray-50">
                            <td className="px-4 py-2">{item.artist}</td>
                            <td className="px-4 py-2">{item.title}</td>
                            <td className="px-4 py-2">{item.media_condition}</td>
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
          ))}
        </div>
      )}
    </div>
  );
};




export default DiscogsKnapsack;

