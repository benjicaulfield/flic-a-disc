import { useState, useEffect } from 'react';
import type { DiscogsRecord } from '../types';
import { apiFetch } from '../api/client';

export default function useRecordTable(endpoint: (page: number) => string) {
  const [records, setRecords] = useState<DiscogsRecord[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    apiFetch(endpoint(page))
      .then(r => r.json())
      .then(data => {
        if (cancelled) return
        setRecords(data.records)
        setTotal(data.total)
      })
      .finally(() => !cancelled && setLoading(false))
    return () => { cancelled = true }
  }, [page, endpoint])

  return { records, total, page, setPage, loading, totalPages: Math.ceil(total / PAGE_SIZE) }
}

