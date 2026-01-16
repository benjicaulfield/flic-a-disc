import { useNavigate } from 'react-router-dom';
import { Paginate } from '../hooks/Paginate';
import { useEbayListings } from '../hooks/useEbayListings';

const EbayBuyItNow = () => {
  const {
    listings,
    loading,
    submitting,
    refreshing,
    error,
    keeperIds,
    completedPages,
    getCachedListings,
    refreshListings,
    toggleLabel,
    submitAnnotations
  } = useEbayListings('/api/ebay/buyitnow', '/api/ebay/refresh_buyitnow');

  const { 
      currentItems: currentPageListings,
      currentPage,
      totalPages,
      nextPage,
      previousPage,
      hasNextPage,
      hasPreviousPage
    } = Paginate(listings, 40);

  const navigate = useNavigate();

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <div className="text-lg text-gray-600">Recommendations...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-white rounded-lg shadow-md p-8 text-center max-w-md">
          <div className="text-red-600 text-xl font-semibold mb-2">Error</div>
          <div className="text-gray-700">{error}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6 mb-6">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold text-slate-700">BUYITNOW</h1>
              <div className="mt-2 text-sm text-slate-500">
                {completedPages} pages completed • {listings.length} similar records found
              </div>
            </div>
            <button
              onClick={refreshListings}
              disabled={refreshing}
              className="mt-4 sm:mt-0 px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:bg-gray-400 text-sm"
            >
              {refreshing ? 'Refreshing...' : '🔄 Refresh'}
            </button>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between">
            <button 
              onClick={previousPage} 
              disabled={!hasPreviousPage}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              ← Previous
            </button>
            
            <span className="text-sm text-gray-700">
              Page <span className="font-medium">{currentPage + 1}</span> of <span className="font-medium">{totalPages}</span>
            </span>
            
            <button 
              onClick={nextPage} 
              disabled={!hasNextPage}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
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
                  <th className="px-2 py-2 text-center font-semibold text-gray-700 w-20 border-r border-gray-200">Keep</th>
                </tr>
              </thead>
              <tbody>
                {currentPageListings.map((listing, index) => {
                  const isKept = keeperIds.has(listing.ebay_id);
                  return (
                    <tr 
                      key={listing.ebay_id} 
                      className={`border-b border-gray-200 hover:bg-gray-50 ${
                        isKept ? 'bg-blue-50' : ''
                      }`}
                    >
                      <td className="px-2 py-2 text-gray-700" title={listing.ebay_title}>
                        {listing.ebay_title || '—'}
                      </td>
                      <td className="px-2 py-1.5 border-r border-gray-200">  {/* ✅ Add this */}
                        ${listing.current_bid || listing.price || '—'}
                      </td>
                      <td className="px-2 py-1.5 text-center border-r border-gray-200">
                        <button
                          onClick={(event) => toggleLabel(listing.id, index, event)}
                          className={`px-2 py-0.5 text-xs rounded border ${
                            isKept
                              ? 'bg-gray-400 text-white border-gray-400'
                              : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-100'
                          }`}
                        >
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
          <button
            onClick={() => navigate('/')}
            className="px-6 py-3 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
          >
            Cancel
          </button>
          
          <button
            onClick={submitAnnotations}
            disabled={submitting}
            className="px-8 py-3 bg-blue-600 text-white rounded disabled:bg-gray-400 hover:bg-blue-700 font-medium"
          >
            {submitting ? 'Submitting...' : `Submit Page (${keeperIds.size} keepers)`}
          </button>
        </div>
      </div>  
    </div> 
  );
};


export default EbayBuyItNow;
