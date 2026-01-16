// hooks/useEbayListings.ts
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { apiFetch, mlFetch } from '../api/client';
import type { BasicEbayListing } from '../types/interfaces';

export const useEbayListings = (endpoint: string, refreshEndpoint: string) => {
  const [listings, setListings] = useState<BasicEbayListing[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedListings, setSelectedListings] = useState<Record<number, boolean>>({});
  const [lastClickedIndex, setLastClickedIndex] = useState<number | null>(null);
  const [completedPages, setCompletedPages] = useState<number>(0);
  const [keeperIds, setKeeperIds] = useState<Set<string>>(new Set());

  const getCachedListings = async () => {
    try {
      setLoading(true);
      const response = await apiFetch(endpoint, {
        credentials: 'include'
      });
      if (response.ok) {
        const data = await response.json();
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

  const refreshListings = async () => {
    setRefreshing(true);
    setError(null);
    try {
      await apiFetch(refreshEndpoint, {
        method: 'POST',
        credentials: 'include'
      });
      await getCachedListings();
    } catch (err) {
      setError('Failed to refresh from eBay');
    } finally {
      setRefreshing(false);
    }
  };

  const toggleLabel = (
    listingId: number, 
    index: number, 
    event: React.MouseEvent,
    currentPageListings: BasicEbayListing[]
  ): void => {
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

  const submitAnnotations = async (currentPageListings: BasicEbayListing[], nextPage: () => void) => {
    try {
      setSubmitting(true);
      const allListings = currentPageListings.map(listing => ({  
        ebay_id: listing.ebay_id,
        label: keeperIds.has(listing.ebay_id)
      }));

      const response = await mlFetch('/ebay/annotated/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ annotations: allListings })
      });

      if (response.ok) {
        const result = await response.json();

        if (result.correct !== undefined && result.total !== undefined) {
          await fetch(`${import.meta.env.VITE_ML_URL}/ebay/batch_performance/`, {
            method: 'POST',
            credentials: 'include',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              correct: result.correct,
              total: result.total
            })
          });
        }

        setCompletedPages(prev => prev + 1);
        nextPage();  
        setKeeperIds(new Set());
        window.scrollTo(0, 0);
      }
    } catch (err) {
      setError('Failed to submit annotations');
    } finally {
      setSubmitting(false);
    }
  };

  return {
    listings,
    setListings,
    loading,
    submitting,
    refreshing,
    error,
    setError,
    selectedListings,
    completedPages,
    keeperIds,
    getCachedListings,
    refreshListings,
    toggleLabel,
    submitAnnotations
  };
};