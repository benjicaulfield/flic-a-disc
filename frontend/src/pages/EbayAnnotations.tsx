import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Paginate } from '../hooks/Paginate';

interface BasicEbayListing {
  id: number;
  ebay_id: string;
  ebay_title: string;
  score: number;
  price?: string;
  current_bid?: string;
  end_date: string;
}

const EbayAnnotation = () => {
  const [listings, setListings] = useState<BasicEbayListing[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedListings, setSelectedListings] = useState<Record<number, boolean>>({});
  const [lastClickedIndex, setLastClickedIndex] = useState<number | null>(null);
  const [completedPages, setCompletedPages] = useState<number>(0);
  const [keeperIds, setKeeperIds] = useState<Set<string>>(new Set());
  const navigate = useNavigate();

  const PAGE_SIZE = 40;

  const {
    currentItems: currentPageListings,
    currentPage, totalPages, nextPage,
    previousPage, hasNextPage, hasPreviousPage
  } = Paginate(listings, PAGE_SIZE);
  
  const loadSimilarListings = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/api/ebay/unannotated', {
        credentials: 'include'
      });
      if (response.ok) {
        const data = await response.json();
        console.log('📊 DATA:', data.listings.length, data.listings[0]); // ✅ Add this
        console.log('Received data:', data);
        console.log('📄 Current page listings:', currentPageListings.length, currentPageListings[0]);

        setListings(data.listings || []);
      } else {
        setError('Failed to load listings');
      }
    } catch (err) {
      setError('Failed to load listings');
    } finally {
      setLoading(false);
    }
  };
  
  useEffect(() => {
    loadSimilarListings();
  }, []);

  const toggleLabel = (listingId: number, index: number, event: React.MouseEvent): void => {
    const listing = currentPageListings[index];
    if (!listing) return;

    if (event.shiftKey && lastClickedIndex !== null) {
      const startIndex = Math.min(lastClickedIndex, index);
      const endIndex = Math.max(lastClickedIndex, index);
      const shouldSelect = !selectedListings[listingId];
      
      setSelectedListings(prev => {
        const newSelected = { ...prev };
        for (let i = startIndex; i <= endIndex; i++) {
          if (currentPageListings[i]) {
            newSelected[currentPageListings[i].id] = shouldSelect;
          }
        }
        return newSelected;
      });

      setKeeperIds(prev => {
        const newSet = new Set(prev);
        for (let i = startIndex; i <= endIndex; i++) {
          if (currentPageListings[i]) {
            if (shouldSelect) {
              newSet.add(currentPageListings[i].ebay_id);
            } else {
              newSet.delete(currentPageListings[i].ebay_id);
            }
          }
        }
        return newSet;
    });
    } else {
      // Regular click: toggle single item
      const newValue = !selectedListings[listingId];
      
      setSelectedListings(prev => ({
        ...prev,
        [listingId]: newValue
      }));
      
      setKeeperIds(prev => {
        const newSet = new Set(prev);
        if (newValue) {
          newSet.add(listing.ebay_id);
        } else {
          newSet.delete(listing.ebay_id);
        }
        return newSet;
      });
    }
  
    setLastClickedIndex(index);
  };

  const submitAnnotations = async () => {
    try {
      setSubmitting(true);
      const allListings = currentPageListings.map(listing => ({  
        ebay_id: listing.ebay_id,
        label: keeperIds.has(listing.ebay_id)
      }));

      const response = await fetch('http://localhost:8001/ml/ebay/annotated/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ annotations: allListings })
      });

      if (response.ok) {
        const result = await response.json();

        if (result.correct !== undefined && result.total !== undefined) {
          await fetch('http://localhost:8001/ml/ebay/batch_performance/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              correct: result.correct,
              total: result.total
            })
          });
        }
      }
      
      setCompletedPages(prev => prev + 1);
      nextPage();  
      setKeeperIds(new Set());
      window.scrollTo(0, 0);
      
    } catch (err) {
      setError('Failed to submit annotations');
    } finally {
      setSubmitting(false);
    }
  };

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
            <h1 className="text-3xl font-bold text-slate-700">Similar Records on eBay</h1>
            <div className="mt-2 sm:mt-0 text-sm text-slate-500">
              {completedPages} pages completed • {listings.length} similar records found
            </div>
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

export default EbayAnnotation;