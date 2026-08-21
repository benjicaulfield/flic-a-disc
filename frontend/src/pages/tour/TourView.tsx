import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import TourDashboard from './stops/TourDashboard';
import TourDiscogs from './stops/TourDiscogs';
import TourEbay from './stops/TourEbay';
import TourKnapsack from './stops/TourKnapsack';

interface TourStop {
  id: string;
  label: string;
  description: string;
  component: React.ComponentType;
}

const TOUR_STOPS: TourStop[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    description: 'The command center. A VS Code-inspired layout showing ML model accuracy over time, a live kanban board for tracking dev tasks, and quick links to every tool in the app.',
    component: TourDashboard,
  },
  {
    id: 'discogs',
    label: 'Discogs Training',
    description: 'The annotation engine. Records from my Discogs wantlist are served up in batches — I mark keepers, the neural bandit learns my taste in real time. Shift-click for range selection.',
    component: TourDiscogs,
  },
  {
    id: 'ebay',
    label: 'eBay Auctions',
    description: 'Tomorrow\'s auctions, pre-filtered by TF-IDF similarity to my Discogs keepers. The model scores each listing and surfaces the ones most likely to be worth bidding on.',
    component: TourEbay,
  },
  {
    id: 'knapsack',
    label: 'Knapsack Solver',
    description: 'Given a Discogs seller and a budget, this solves the 0/1 knapsack problem to find the optimal subset of records — maximizing the model\'s predicted value within your spend limit.',
    component: TourKnapsack,
  },
];

export default function TourView() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const navigate = useNavigate();

  const currentStop = TOUR_STOPS[currentIndex];
  const StopComponent = currentStop.component;
  const isFirst = currentIndex === 0;
  const isLast = currentIndex === TOUR_STOPS.length - 1;

  return (
    <div className="relative min-h-screen">

      {/* Tour overlay — sits on top, fully interactive */}
      <div className="fixed top-0 left-0 right-0 z-50">
        <div className="bg-black/80 backdrop-blur-sm text-white px-6 py-4">
          <div className="max-w-5xl mx-auto">
            {/* Progress dots */}
            <div className="flex justify-center gap-2 mb-3">
              {TOUR_STOPS.map((stop, i) => (
                <button
                  key={stop.id}
                  onClick={() => setCurrentIndex(i)}
                  className={`w-2 h-2 rounded-full transition-all ${
                    i === currentIndex
                      ? 'bg-white w-6'
                      : 'bg-white/40 hover:bg-white/70'
                  }`}
                  aria-label={`Go to ${stop.label}`}
                />
              ))}
            </div>

            {/* Stop info + navigation */}
            <div className="flex items-center gap-4">
              <button
                onClick={() => setCurrentIndex(i => i - 1)}
                disabled={isFirst}
                className="px-4 py-2 rounded border border-white/30 text-sm font-medium
                           hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed
                           transition-colors shrink-0"
              >
                ← Prev
              </button>

              <div className="flex-1 min-w-0">
                <div className="flex items-baseline gap-3 mb-1">
                  <span className="text-xs font-semibold uppercase tracking-widest text-white/50">
                    {currentIndex + 1} / {TOUR_STOPS.length}
                  </span>
                  <span className="text-sm font-bold text-white">
                    {currentStop.label}
                  </span>
                </div>
                <p className="text-xs text-white/70 leading-relaxed line-clamp-2">
                  {currentStop.description}
                </p>
              </div>

              {isLast ? (
                <button
                  onClick={() => navigate('/')}
                  className="px-4 py-2 rounded bg-white text-black text-sm font-semibold
                             hover:bg-white/90 transition-colors shrink-0"
                >
                  Back to site →
                </button>
              ) : (
                <button
                  onClick={() => setCurrentIndex(i => i + 1)}
                  className="px-4 py-2 rounded border border-white/30 text-sm font-medium
                             hover:bg-white/10 transition-colors shrink-0"
                >
                  Next →
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Gradient fade */}
        <div className="h-8 bg-gradient-to-b from-black/60 to-transparent" />
      </div>

      {/* Page content — frozen, no interaction */}
      <div
        style={{ pointerEvents: 'none', userSelect: 'none' }}
        className="min-h-screen pt-32"
      >
        <StopComponent />
      </div>

      {/* Exit button — top right */}
      <button
        onClick={() => navigate('/')}
        className="fixed top-4 right-4 z-50 px-3 py-1.5 text-xs font-medium
                   bg-black/60 text-white/80 rounded border border-white/20
                   hover:bg-black/80 hover:text-white transition-colors backdrop-blur-sm"
      >
        ✕ Exit tour
      </button>
    </div>
  );
}