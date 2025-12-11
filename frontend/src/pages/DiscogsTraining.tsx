import { useEffect, useState, Fragment } from 'react';
import type { DiscogsRecord, MLData, PerformanceStats } from '../types/interfaces';
import { Paginate } from '../hooks/Paginate';
import { apiFetch } from "../api/client";

const DiscogsTraining = () => {
  const [records, setRecords] = useState<DiscogsRecord[]>([]);
  const [labeledCount, setLabeledCount] = useState<number>(0);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [selectedRecords, setSelectedRecords] = useState<Record<number, boolean>>({});
  const [lastClickedIndex, setLastClickedIndex] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [saving, setSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [mlData, setMlData] = useState<MLData | null>(null);
  const [cumulativeStats, setCumulativeStats] = useState<PerformanceStats | null>(null);

  const PAGE_SIZE = 20;


  const {
      currentItems: currentPageRecords,
      currentPage, totalPages, nextPage,
      previousPage, hasNextPage, hasPreviousPage
    } = Paginate(records, PAGE_SIZE)

  useEffect(() => {
    loadPage();
    fetchStats();
  }, []);

  const loadPage = async () => {
    try {
      setLoading(true);
      const response = await apiFetch("/api/discogs/keepers", {
        credentials: 'include',
      });
          
      if (response.ok) {
          const data = await response.json();
          console.log('First record:', data.records?.[0]);  // ← ADD THIS LINE

          setRecords(data.records)
          setMlData({
            predictions: data.predictions,
            mean_predictions: data.mean_predictions,
            threshold: 0.75,
            uncertainties: data.uncertainties,
            model_version: data.model_version
          });
      } else {
        setError('Failed to load listings');
      }
    } catch (err) {
      setError('Failed to load listings');
    } finally {
      setLoading(false);
    }
  }

  const fetchStats = async () => {
    try {
      const response = await apiFetch("/api/discogs/stats", {
        credentials: 'include',
      });


      if (response.ok) {
        const stats = await response.json();
        setLabeledCount(stats.labeled);
        setTotalCount(stats.total);

      } else {
        console.log('Response not ok', response.status);
      }
    } catch (err) {
      console.error('error somewhere', err);
    }
  };

  const toggleLabel = (recordId: number, index: number, event: React.MouseEvent): void => {
    if (event.shiftKey && lastClickedIndex !== null) {
      // Shift+click: select range
      const startIndex = Math.min(lastClickedIndex, index);
      const endIndex = Math.max(lastClickedIndex, index);
      
      // Determine what state to apply - if the current item is unselected, select the range
      const shouldSelect = !selectedRecords[recordId];
      
      setSelectedRecords(prev => {
        const newSelected = { ...prev };
        for (let i = startIndex; i <= endIndex; i++) {
          if (currentPageRecords[i]) {
            newSelected[currentPageRecords[i].id] = shouldSelect;
          }
        }
        return newSelected;
      });
    } else {
      // Regular click: toggle single item
      setSelectedRecords(prev => ({
        ...prev,
        [recordId]: !prev[recordId]
      }));
    }
    
    setLastClickedIndex(index);
  };

  const savePage = async () => {
    console.log("save page called");
    try {
      setSaving(true);
      const labels = currentPageRecords.map(listing => ({
        id: listing.id,
        label: selectedRecords[listing.id] || false
      }));

      const mlRecords = currentPageRecords.map(record => ({
        listing_id: record.id,
        artist: record.artist,
        title: record.title,
        label: record.label,
        genres: record.genres,
        styles: record.styles,
        wants: record.wants,
        haves: record.haves,
        year: record.year,
      }));

      // ✅ WAIT for labels to be saved (this creates the BatchPerformance record)
      await apiFetch("/api/discogs/labels", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ 
          labels, 
          records: mlRecords, 
          predictions: mlData?.predictions || [],
          mean_predictions: mlData?.mean_predictions || [],
          uncertainties: mlData?.uncertainties || []
        })
      });

      setLabeledCount(prev => prev + currentPageRecords.length);
      setShowResults(true);
      
      const { divider } = sortListingsByAgreement();
      
      // ✅ NOW fetch performance (BatchPerformance record exists)
      const response = await apiFetch('/api/discogs/performance', {
        method: 'POST',
        headers: {'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          correct: divider,
          total: currentPageRecords.length
        })
      });

      const perfData = await response.json();
      console.log('📊 Performance data:', perfData);  // ✅ ADD THIS DEBUG
      setCumulativeStats(perfData);

    } catch (err) {
      console.error('Save error:', err);  // ✅ ADD THIS DEBUG
      setError('Failed to save labels');
    } finally {
      setSaving(false);
    }
  };

  const sortListingsByAgreement = () => {
    if (!mlData) {
      return { listings: currentPageRecords, divider: 0, accuracy: 0 };
    }

    const categorized = currentPageRecords.map((listing, index) => {
      const userSelected = selectedRecords[listing.id] || false;
      // Calculate binary decision from mean_predictions right here
      const modelSelected = mlData.mean_predictions[index] > 0.5;
      const agreement = userSelected === modelSelected;
      
      return { listing, userSelected, modelSelected, agreement };
    });

    const agreements = categorized.filter(item => item.agreement);
    const disagreements = categorized.filter(item => !item.agreement);
    const sorted = [...agreements, ...disagreements];
    const accuracy = (agreements.length / categorized.length) * 100;

    return {
      listings: sorted.map(item => item.listing),
      divider: agreements.length,
      accuracy: Math.round(accuracy)
    };
  };

  const loadNextBatch = async () => {
    setShowResults(false);
    setSelectedRecords({});
    setLastClickedIndex(null);
    setMlData(null);
    await loadPage();
  };
  

  const { listings: displayListings, divider, accuracy } = showResults 
    ? sortListingsByAgreement() 
    : { listings: currentPageRecords, divider: 0, accuracy: 0 };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <div className="text-lg text-gray-600">Loading listings...</div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-white rounded-lg shadow-md p-8 text-center max-w-md">
          <div className="text-red-600 text-xl font-semibold mb-2">Error</div>
          <div className="text-gray-700">{error}</div>
        </div>
      </div>
    )
  }

  if (records.length === 0) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-white rounded-lg shadow-md p-12 text-center">
          <div className="text-3xl font-bold text-gray-800 mb-4">🎉 YOU ARE DONE! 🎉</div>
          <div className="text-gray-600">All records have been evaluated</div>
        </div>
      </div>
    )
  }

  return (
    

    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4">
          <h1 className="text-3xl font-bold text-gray-900">
            {showResults ? 'Prediction Results' : 'Discogs Keepers'}
          </h1>              
          <div className="mt-2 sm:mt-0 text-sm text-gray-600">
            Total labeled: <span className="font-semibold text-blue-600">{labeledCount}/{totalCount}</span>
          </div>
        </div>

        {showResults && (
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="text-center">
                <div className="text-sm text-blue-700 font-medium">This Batch</div>
                <div className={`text-4xl font-bold mt-2 ${
                  (accuracy ?? 0) >= 80 ? 'text-green-600' : (accuracy ?? 0) >= 70 ? 'text-yellow-600' : 'text-red-600'
                }`}>
                  {accuracy ?? 0}%
                </div>
                <div className="text-sm text-blue-600 mt-1">
                  {divider} correct out of {currentPageRecords.length} predictions
                </div>
              </div>
            </div>
            
            {cumulativeStats && (
              <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
                <div className="text-center">
                  <div className="text-sm text-purple-700 font-medium">
                    Rolling Window (Last {cumulativeStats?.total_batches ?? 0} Batches)
                  </div>
                  <div className={`text-4xl font-bold mt-2 ${
                    (cumulativeStats?.cumulative_accuracy ?? 0) >= 80 ? 'text-green-600' : 
                    (cumulativeStats?.cumulative_accuracy ?? 0) >= 70 ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                    {Math.round(cumulativeStats?.cumulative_accuracy ?? 0)}%
                  </div>
                  <div className="text-sm text-purple-600 mt-1">
                    up to 100 batch rolling average
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Pagination Controls */}
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
          <table className="w-full table-fixed divide-y divide-gray-200 text-xs">
            <thead className="bg-gray-50">
              <tr>
                <th className="w-12 px-2 py-2 text-left font-medium text-gray-500 uppercase">✓</th>
                <th className="w-32 px-2 py-2 text-left font-medium text-gray-500 uppercase">Artist</th>
                <th className="w-32 px-2 py-2 text-left font-medium text-gray-500 uppercase">Title</th>
                <th className="w-12 px-2 py-2 text-left font-medium text-gray-500 uppercase">Year</th>
                <th className="w-28 px-2 py-2 text-left font-medium text-gray-500 uppercase">Label</th>
                <th className="w-14 px-2 py-2 text-center font-medium text-gray-500 uppercase">Want</th>
                <th className="w-14 px-2 py-2 text-center font-medium text-gray-500 uppercase">Have</th>
                <th className="w-20 px-2 py-2 text-left font-medium text-gray-500 uppercase">Genre</th>
                <th className="w-20 px-2 py-2 text-left font-medium text-gray-500 uppercase">Style</th>
                <th className="w-20 px-2 py-2 text-right font-medium text-gray-500 uppercase">Sugg</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {displayListings.map((record, index) => {
                const dividingLine = showResults && index === divider;

                return (
                  <Fragment key={record.id}>
                    {dividingLine && (
                      <tr>
                        <td colSpan={12} className="px-0 py-0">
                          <div className="bg-red-50 px-4 py-2 border-t-4 border-b border-red-200">
                            <h2 className="text-sm font-semibold text-red-800">
                              ✗ Incorrect Predictions
                            </h2>
                          </div>
                        </td>
                      </tr>
                    )}
                  <tr className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-2 py-2">
                      <button 
                        className={`w-6 h-6 rounded border flex items-center justify-center text-xs transition-colors ${
                          selectedRecords[record.id] 
                            ? 'bg-green-500 border-green-500 text-white hover:bg-green-600' 
                            : 'bg-white border-gray-300 hover:border-gray-400 hover:bg-gray-50'
                        }`}
                        onClick={(event) => toggleLabel(record.id, index, event)}
                      >
                        {selectedRecords[record.id] && '✓'}
                      </button>
                    </td>
                    <td className="px-2 py-2 font-medium text-gray-900 truncate" title={record.artist}>
                      {record.artist}
                    </td>
                    <td className="px-2 py-2 text-gray-700 truncate" title={record.title}>
                      {record.title}
                    </td>
                    <td className="px-2 py-2 text-gray-700">
                      {record.year}
                    </td>
                    <td className="px-2 py-2 text-gray-700 truncate" title={record.label}>
                      {record.label}
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
                    <td className="px-2 py-2 text-gray-500 text-right truncate">
                      {record.suggested_price 
                        ? `$${parseFloat(record.suggested_price.replace(/[^0-9.]/g, '')).toFixed(2)}` 
                        : '—'}
                    </td>
                  </tr>
                </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mt-6 flex justify-center">
      {showResults ? (
        <button
          className="px-8 py-3 rounded-lg text-white font-medium bg-blue-600 hover:bg-blue-700 transition-colors"            
          onClick={loadNextBatch}
        >
          Continue to Next Batch
        </button>
      ) : (
        <button
          className={`px-8 py-3 rounded-lg text-white font-medium transition-colors ${
            saving 
              ? 'bg-gray-400 cursor-not-allowed' 
              : 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800'
          }`}
          onClick={savePage}
          disabled={saving}
        >
          {saving ? (
            <div className="flex items-center">
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2"></div>
              Submitting...
            </div>
          ) : (
            'Submit Page'
          )}
        </button>
      )}
    </div>
  </div>
)};

export default DiscogsTraining;
