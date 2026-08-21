const mockAuctions = [
  { id: 1, ebay_title: 'ARTHUR RUSSELL World Of Echo LP Rough Trade 1986 EX/VG+', current_bid: '28.50', tfidf_score: 0.847, kept: false },
  { id: 2, ebay_title: 'BROADCAST Haha Sound 2xLP Warp Records 2003 NM/NM', current_bid: '22.00', tfidf_score: 0.821, kept: true },
  { id: 3, ebay_title: 'GROUPER Dragging A Dead Deer Up A Hill LP Type Records NM', current_bid: '41.00', tfidf_score: 0.803, kept: true },
  { id: 4, ebay_title: 'FENNESZ Endless Summer LP Mego 2001 EX original press', current_bid: '55.00', tfidf_score: 0.791, kept: false },
  { id: 5, ebay_title: 'YELLOW SWANS At All Ends LP Not Not Fun 2008 VG+/NM', current_bid: '38.00', tfidf_score: 0.778, kept: true },
  { id: 6, ebay_title: 'OREN AMBARCHI Hubris 2xLP Editions Mego 2016 NM', current_bid: '19.99', tfidf_score: 0.762, kept: false },
  { id: 7, ebay_title: 'PAULINE OLIVEROS Deep Listening LP New Albion sealed', current_bid: '44.00', tfidf_score: 0.745, kept: false },
  { id: 8, ebay_title: 'THE DEAD C Trapdoor Fucking Exit LP Siltbreeze orig EX', current_bid: '88.00', tfidf_score: 0.738, kept: true },
  { id: 9, ebay_title: 'BILL ORCUTT A History Of Every One LP Palilalia NM/NM', current_bid: '32.00', tfidf_score: 0.721, kept: false },
  { id: 10, ebay_title: 'JANDEK Ready For The House LP Corwood VG+ rare', current_bid: '75.00', tfidf_score: 0.714, kept: false },
  { id: 11, ebay_title: 'LOREN CONNORS Blues LP Temporary Residence 1995 EX', current_bid: '48.00', tfidf_score: 0.698, kept: true },
  { id: 12, ebay_title: 'THE NECKS Drive By LP ReR Megacorp 2018 NM/NM', current_bid: '26.00', tfidf_score: 0.682, kept: false },
  { id: 13, ebay_title: 'MATMOS The Rose Has Teeth LP Matador Records NM', current_bid: '17.50', tfidf_score: 0.671, kept: false },
  { id: 14, ebay_title: 'VAMPIRE BELT Vampire Belt LP Drag City 2006 EX', current_bid: '22.00', tfidf_score: 0.659, kept: true },
  { id: 15, ebay_title: 'DUCKTAILS III Arcade Dynamics LP Woodsist NM', current_bid: '14.00', tfidf_score: 0.644, kept: false },
];

export default function TourEbay() {
  const keptIds = new Set(mockAuctions.filter(a => a.kept).map(a => a.id));

  return (
    <div className="min-h-screen bg-slate-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6 mb-6">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold text-slate-700">TOMORROW'S AUCTIONS</h1>
              <div className="mt-2 text-sm text-slate-500">
                0 pages completed • {mockAuctions.length} similar records found
              </div>
            </div>
            <button className="mt-4 sm:mt-0 px-4 py-2 bg-green-600 text-white rounded text-sm">
              🔄 Refresh
            </button>
          </div>

          <div className="flex items-center justify-between">
            <button className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md opacity-50 cursor-not-allowed">
              ← Previous
            </button>
            <span className="text-sm text-gray-700">Page <span className="font-medium">1</span> of <span className="font-medium">1</span></span>
            <button className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md opacity-50 cursor-not-allowed">
              Next →
            </button>
          </div>
        </div>

        {/* Table */}
        <div className="bg-white rounded-lg shadow-sm overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead className="bg-slate-100 border-b border-slate-200">
                <tr>
                  <th className="px-2 py-2 text-left font-semibold text-gray-700 border-r border-gray-200">Title</th>
                  <th className="px-2 py-2 text-left font-semibold text-gray-700 w-16 border-r border-gray-200">Price</th>
                  <th className="px-2 py-2 text-left font-semibold text-gray-700 w-16 border-r border-gray-200">Score</th>
                  <th className="px-2 py-2 text-center font-semibold text-gray-700 w-20 border-r border-gray-200">Keep</th>
                </tr>
              </thead>
              <tbody>
                {mockAuctions.map((auction) => {
                  const isKept = keptIds.has(auction.id);
                  return (
                    <tr
                      key={auction.id}
                      className={`border-b border-gray-200 ${isKept ? 'bg-blue-50' : 'hover:bg-gray-50'}`}
                    >
                      <td className="px-2 py-2 text-gray-700">{auction.ebay_title}</td>
                      <td className="px-2 py-1.5 border-r border-gray-200">${auction.current_bid}</td>
                      <td className="px-2 py-1.5 border-r border-gray-200">{auction.tfidf_score.toFixed(3)}</td>
                      <td className="px-2 py-1.5 text-center border-r border-gray-200">
                        <button className={`px-2 py-0.5 text-xs rounded border ${
                          isKept
                            ? 'bg-gray-400 text-white border-gray-400'
                            : 'bg-white text-gray-700 border-gray-300'
                        }`}>
                          {isKept ? '✓' : 'Keep'}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        <div className="mt-6 flex justify-end gap-4">
          <button className="px-6 py-3 bg-gray-200 text-gray-700 rounded">Cancel</button>
          <button className="px-8 py-3 bg-blue-600 text-white rounded font-medium">
            Submit Page ({keptIds.size} keepers)
          </button>
        </div>
      </div>
    </div>
  );
}
