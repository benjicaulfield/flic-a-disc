import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { apiFetch } from '../api/client';

import type { EbayListing } from '../types/interfaces';


const EbayAuctions = () => {
  const [listings, setListings] = useState<EbayListing[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(0);
  const navigate = useNavigate();

  const PAGE_SIZE = 40;
  const startIdx = currentPage * PAGE_SIZE;
  const endIdx = startIdx + PAGE_SIZE;
  const currentPageListings = listings.slice(startIdx, endIdx);
  const totalPages = Math.ceil(listings.length / PAGE_SIZE);

  useEffect(() => {
    loadListings();
  }, []);

  const loadListings = async () => {
    setLoading(true);
    try {
      console.log('fetching auctions from api/ebay/auctions');
      const response = await apiFetch('/api/ebay/auctions', {
        credentials: 'include',
      });
      const data = await response.json();
      console.log('Keys in first listing:', Object.keys(data.listings[0])); // ← This will show us the exact property names
      setListings(data.listings);
    } catch (err) {
      setError('Failed to load auctions');
    } finally {
      setLoading(false);
    }
  };

  const toggleSelection = (ebayId: string) => {
    setSelectedIds(prev => {
      const newSet = new Set(prev);
      if (newSet.has(ebayId)) {
        newSet.delete(ebayId);
      } else {
        newSet.add(ebayId);
      }
      return newSet;
    });
  };

  const saveSelected = async () => {
    if (selectedIds.size === 0) return;
    
    setSaving(true);
    try {
      await apiFetch('/api/ebay/save', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ebay_ids: Array.from(selectedIds) })
      });
      navigate('/ebay/saved');
    } catch (err) {
      setError('Failed to save listings');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <div className="text-lg text-gray-600">Loading auctions...</div>
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
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <div className="flex justify-between items-center mb-4">
            <h1 className="text-3xl font-bold text-gray-900">eBay Auctions Ending Soon</h1>
            <button
              onClick={() => navigate('/ebay/saved')}
              className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
            >
              View Saved
            </button>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between">
            <button 
              onClick={() => setCurrentPage(p => p - 1)} 
              disabled={currentPage === 0}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              ← Previous
            </button>
            
            <span className="text-sm text-gray-700">
              Page <span className="font-medium">{currentPage + 1}</span> of <span className="font-medium">{totalPages}</span>
              {' '}({listings.length} total auctions)
            </span>
            
            <button 
              onClick={() => setCurrentPage(p => p + 1)} 
              disabled={currentPage >= totalPages - 1}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next →
            </button>
          </div>
        </div>

        {/* Table */}
        <div className="bg-white rounded-lg shadow-sm overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left font-medium text-gray-500">Save</th>
                <th className="px-4 py-3 text-left font-medium text-gray-500">Title</th>
                <th className="px-4 py-3 text-left font-medium text-gray-500">Current Bid</th>
                <th className="px-4 py-3 text-left font-medium text-gray-500">Bids</th>
                <th className="px-4 py-3 text-left font-medium text-gray-500">Ends</th>
                <th className="px-4 py-3 text-left font-medium text-gray-500">Link</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {currentPageListings.map((listing) => (
                <tr key={listing.ebay_id} className="hover:bg-gray-50">
                  <td className="px-4 py-3">
                    <button
                      onClick={() => toggleSelection(listing.ebay_id.toString())}
                      className={`w-6 h-6 rounded border flex items-center justify-center ${
                        selectedIds.has(listing.ebay_id.toString())
                          ? 'bg-blue-500 border-blue-500 text-white'
                          : 'border-gray-300'
                      }`}
                    >
                      {selectedIds.has(listing.ebay_id.toString()) && '✓'}
                    </button>
                  </td>
                  <td className="px-4 py-3 max-w-md truncate" title={listing.title}>
                    {listing.title}
                  </td>
                  <td className="px-4 py-3">
                    ${listing.current_bid || listing.price}
                  </td>
                  <td className="px-4 py-3">{listing.bid_count}</td>
                  <td className="px-4 py-3">
                    {new Date(listing.end_date).toLocaleDateString()} {new Date(listing.end_date).toLocaleTimeString()}
                  </td>
                  <td className="px-4 py-3">
                    <a href={`https://www.ebay.com/itm/${listing.ebay_id}`} target="_blank" className="text-blue-600 hover:underline">
                      View
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Bottom Actions */}
        <div className="mt-6 flex justify-between">
          <button
            onClick={() => navigate('/ebay/saved')}
            className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
          >
            View Saved Listings
          </button>
          
          <button
            onClick={saveSelected}
            disabled={selectedIds.size === 0 || saving}
            className="px-8 py-3 bg-blue-600 text-white rounded disabled:bg-gray-400 hover:bg-blue-700"
          >
            {saving ? 'Saving...' : `Save Selected (${selectedIds.size})`}
          </button>
        </div>
      </div>
    </div>
  );
};

export default EbayAuctions;
