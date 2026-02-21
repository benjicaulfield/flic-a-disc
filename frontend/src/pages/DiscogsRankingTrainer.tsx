import { useState, useEffect } from 'react';
import type { DiscogsRecord } from '../types/interfaces';
import { apiFetch } from "../api/client";

const DiscogsRankingTrainer = () => {
  const [records, setRecords] = useState<DiscogsRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [batchIndex, setBatchIndex] = useState(0);
  const [totalBatches, setTotalBatches] = useState(0);
  const [draggedId, setDraggedId] = useState<number | null>(null);

  useEffect(() => {
    loadBatch();
  }, []);

  const loadBatch = async () => {
    setLoading(true);
    try {
      const response = await apiFetch('/api/ranking/batch', {
        credentials: 'include'
      });
      
      if (response.ok) {
        const data = await response.json();
        setRecords(data.records);
        setBatchIndex(data.batch_index);
        setTotalBatches(data.total_batches);
      }
    } catch (err) {
      console.error('Failed to load batch:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDragStart = (e: React.DragEvent, recordId: number) => {
    setDraggedId(recordId);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  };

  const handleDrop = (e: React.DragEvent, targetId: number) => {
    e.preventDefault();
    if (draggedId && draggedId !== targetId) {
      const draggedIndex = records.findIndex(r => r.id === draggedId);
      const targetIndex = records.findIndex(r => r.id === targetId);
      
      const newRecords = [...records];
      const [removed] = newRecords.splice(draggedIndex, 1);
      newRecords.splice(targetIndex, 0, removed);
      
      setRecords(newRecords);
    }
    setDraggedId(null);
  };

  const submitRanking = async () => {
    setSubmitting(true);
    try {
      const ranking = records.map(r => r.id);
      
      await apiFetch('/api/ranking/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ ranking })
      });
      
      await loadBatch();
    } catch (err) {
      console.error('Failed to submit ranking:', err);
    } finally {
      setSubmitting(false);
    }
  };

  if (loading && records.length === 0) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-lg text-gray-600">Loading batch...</div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Ranking Trainer</h1>
        <p className="text-gray-600 mb-4">
          Drag records to rank them from most to least preferred
        </p>
        <div className="text-sm text-gray-500">
          Batch {batchIndex + 1} of {totalBatches}
        </div>
      </div>

      <div className="space-y-2">
        {records.map((record, index) => (
          <div
            key={record.id}
            draggable
            onDragStart={(e) => handleDragStart(e, record.id)}
            onDragOver={handleDragOver}
            onDrop={(e) => handleDrop(e, record.id)}
            className={`bg-white border rounded-lg p-4 cursor-move hover:shadow-md transition-shadow ${
              draggedId === record.id ? 'opacity-50' : 'opacity-100'
            } border-gray-200 hover:border-blue-400`}
          >
            <div className="flex items-center gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center font-bold text-blue-700">
                {index + 1}
              </div>
              <div className="flex-1 min-w-0">
                <div className="font-semibold text-gray-900 truncate">{record.artist}</div>
                <div className="text-sm text-gray-600 truncate">{record.title}</div>
                <div className="text-xs text-gray-500 mt-1">
                  {record.label} • {record.year} • {record.genres[0]}
                </div>
              </div>
              <div className="flex-shrink-0 text-right">
                <div className="text-xs text-gray-500">Want: {record.wants}</div>
                <div className="text-xs text-gray-500">Have: {record.haves}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 flex justify-center">
        <button
          onClick={submitRanking}
          disabled={submitting}
          className={`px-8 py-3 rounded-lg text-white font-medium transition-colors ${
            submitting 
              ? 'bg-gray-400 cursor-not-allowed' 
              : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {submitting ? 'Submitting...' : 'Submit Ranking'}
        </button>
      </div>
    </div>
  );
};

export default DiscogsRankingTrainer;