import { useEffect, useState } from 'react';
import { Paginate } from '../hooks/Paginate';
import { apiFetch } from '../api/client';
import type { EbayListing } from '../types/interfaces';

const EbayKeepers = () => {
  const [listings, setListings] = useState<EbayListing[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedListings, setSelectedListings] = useState<Record<number, boolean>>({});
  const [lastClickedIndex, setLastClickedIndex] = useState<number | null>(null);
  const [completedPages, setCompletedPages] = useState<number>(0);

  const PAGE_SIZE = 40;

  const {
    currentItems: currentPageListings,
    currentPage, totalPages, nextPage,
    previousPage, hasNextPage, hasPreviousPage
  } = Paginate(listings, PAGE_SIZE);

  const loadListings = async () => {
    try {
      setLoading(true);
      const response = await apiFetch('/api/ebay/annotate_keepers', {
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

  useEffect(() => {
    loadListings();
  }, []);
}

export default EbayKeepers;