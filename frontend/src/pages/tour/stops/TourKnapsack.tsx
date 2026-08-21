const mockResults = {
  seller: 'dustygrooves',
  total_cost: 187.50,
  total_selected: 5,
  total_score: 4.312,
  knapsack: [
    { artist: 'Arthur Russell', title: 'World Of Echo', media_condition: 'VG+', wants: 4821, haves: 1203, suggested_price: 42.00, price: 34.00, currency: 'USD', score: 0.934 },
    { artist: 'Grouper', title: 'Dragging A Dead Deer', media_condition: 'NM', wants: 5930, haves: 2100, suggested_price: 55.00, price: 48.00, currency: 'USD', score: 0.911 },
    { artist: 'Broadcast', title: 'Haha Sound', media_condition: 'NM', wants: 3102, haves: 890, suggested_price: 28.00, price: 22.00, currency: 'USD', score: 0.882 },
    { artist: 'Yellow Swans', title: 'At All Ends', media_condition: 'VG+', wants: 1100, haves: 289, suggested_price: 70.00, price: 55.50, currency: 'USD', score: 0.861 },
    { artist: 'Loren Connors', title: 'Blues', media_condition: 'EX', wants: 720, haves: 188, suggested_price: 65.00, price: 28.00, currency: 'USD', score: 0.724 },
  ],
  contenders: [
    { artist: 'Fennesz', title: 'Endless Summer', media_condition: 'EX', wants: 6204, haves: 1800, suggested_price: 48.00, price: 38.00, currency: 'USD', score: 0.698 },
    { artist: 'The Dead C', title: 'Trapdoor Fucking Exit', media_condition: 'VG+', wants: 980, haves: 220, suggested_price: 120.00, price: 95.00, currency: 'USD', score: 0.671 },
  ],
};

export default function TourKnapsack() {
  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">KNAPSACK PROBLEM</h1>

      {/* Form - pre-filled */}
      <form className="max-w-md bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
        <div className="mb-6">
          <label className="block text-gray-700 text-sm font-bold mb-2">Budget ($)</label>
          <input
            readOnly
            value="200"
            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight"
          />
        </div>
        <div className="mb-6">
          <label className="block text-gray-700 text-sm font-bold mb-2">Seller</label>
          <input
            readOnly
            value="dustygrooves"
            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight"
          />
        </div>
        <button className="bg-blue-500 text-white font-bold py-2 px-4 rounded opacity-60">
          lets do this....
        </button>
      </form>

      {/* Results */}
      <div className="mt-6">
        <h2 className="text-2xl font-bold mb-4">
          Results for {mockResults.seller} — ${mockResults.total_cost.toFixed(2)} / $200.00
        </h2>

        <div className="bg-white shadow-md rounded p-6">
          {/* Summary */}
          <div className="grid grid-cols-3 gap-4 mb-6 bg-gray-50 p-4 rounded">
            <div>
              <p className="text-sm text-gray-600">Items Selected</p>
              <p className="text-2xl font-bold">{mockResults.total_selected}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Total Cost</p>
              <p className="text-2xl font-bold">${mockResults.total_cost.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Total Score</p>
              <p className="text-2xl font-bold">{mockResults.total_score.toFixed(3)}</p>
            </div>
          </div>

          {/* Selected Items */}
          <div className="mb-6">
            <h4 className="font-bold text-lg mb-2">🎯 Selected Items</h4>
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead className="bg-green-100">
                  <tr>
                    {['Artist', 'Title', 'Condition', 'Wants', 'Haves', 'Sugg. Price', 'Price', 'Score'].map(h => (
                      <th key={h} className="px-4 py-2 text-left text-sm">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {mockResults.knapsack.map((item, i) => (
                    <tr key={i} className="border-b hover:bg-gray-50">
                      <td className="px-4 py-2">{item.artist}</td>
                      <td className="px-4 py-2">{item.title}</td>
                      <td className="px-4 py-2">{item.media_condition}</td>
                      <td className="px-4 py-2">{item.wants.toLocaleString()}</td>
                      <td className="px-4 py-2">{item.haves.toLocaleString()}</td>
                      <td className="px-4 py-2">${item.suggested_price.toFixed(2)}</td>
                      <td className="px-4 py-2">${item.price.toFixed(2)} {item.currency}</td>
                      <td className="px-4 py-2">{item.score.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Contenders */}
          <div>
            <h4 className="font-bold text-lg mb-2">📋 Top Contenders (Not Selected)</h4>
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead className="bg-gray-100">
                  <tr>
                    {['Artist', 'Title', 'Condition', 'Wants', 'Haves', 'Sugg. Price', 'Price', 'Score'].map(h => (
                      <th key={h} className="px-4 py-2 text-left text-sm">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {mockResults.contenders.map((item, i) => (
                    <tr key={i} className="border-b hover:bg-gray-50">
                      <td className="px-4 py-2">{item.artist}</td>
                      <td className="px-4 py-2">{item.title}</td>
                      <td className="px-4 py-2">{item.media_condition}</td>
                      <td className="px-4 py-2">{item.wants.toLocaleString()}</td>
                      <td className="px-4 py-2">{item.haves.toLocaleString()}</td>
                      <td className="px-4 py-2">${item.suggested_price.toFixed(2)}</td>
                      <td className="px-4 py-2">${item.price.toFixed(2)} {item.currency}</td>
                      <td className="px-4 py-2">{item.score.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}