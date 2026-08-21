import { useEffect, useMemo, useState } from 'react';
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell,
} from 'recharts';
import RagLayout from './RagLayout';
import { getTimeline } from './api';
import type { TimelineEvent } from './api';

const CATEGORY_COLORS: Record<string, string> = {
  feature: '#4ec9b0',
  bugfix: '#f48771',
  infra: '#007acc',
  decision: '#dcdcaa',
  pivot: '#c586c0',
};

const CATEGORY_Y: Record<string, number> = {
  feature: 5,
  infra: 4,
  decision: 3,
  bugfix: 2,
  pivot: 1,
};

export default function RagTimeline() {
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setBusy(true);
    setError(null);
    try {
      const res = await getTimeline();
      setEvents(res.events);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const chartData = useMemo(
    () =>
      events.map((e, i) => ({
        ...e,
        x: new Date(e.date).getTime(),
        y: CATEGORY_Y[e.category] ?? 3,
        idx: i,
      })),
    [events],
  );

  const exportJson = () => {
    const blob = new Blob([JSON.stringify(events, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'timeline.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <RagLayout>
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-white">project trajectory</h2>
          <div className="flex gap-2">
            <button
              onClick={load}
              disabled={busy}
              className="px-3 py-1 text-xs text-[#858585] hover:text-[#d4d4d4] disabled:opacity-50"
            >
              {busy ? 'building…' : 'rebuild'}
            </button>
            {events.length > 0 && (
              <button
                onClick={exportJson}
                className="px-3 py-1 text-xs bg-[#4ec9b0] text-[#1e1e1e] rounded hover:bg-[#6ed9c3]"
              >
                export json
              </button>
            )}
          </div>
        </div>

        {error && (
          <div className="bg-[#5a1d1d] border border-[#f48771] rounded p-3 text-xs text-[#f48771]">
            {error}
          </div>
        )}

        {busy && events.length === 0 ? (
          <div className="flex items-center justify-center py-20">
            <div className="animate-spin rounded-full h-12 w-12 border-2 border-[#007acc] border-t-transparent" />
          </div>
        ) : events.length === 0 ? (
          <div className="text-center py-20 text-[#6a6a6a] text-sm">
            No dated conversations ingested yet.
          </div>
        ) : (
          <>
            {/* scatter chart */}
            <div className="bg-[#252526] border border-[#3e3e42] rounded p-5">
              <ResponsiveContainer width="100%" height={280}>
                <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#3e3e42" />
                  <XAxis
                    dataKey="x"
                    type="number"
                    domain={['dataMin', 'dataMax']}
                    tickFormatter={(ts) =>
                      new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
                    }
                    stroke="#858585"
                    style={{ fontSize: '10px' }}
                  />
                  <YAxis
                    dataKey="y"
                    type="number"
                    domain={[0, 6]}
                    ticks={[1, 2, 3, 4, 5]}
                    tickFormatter={(v) =>
                      Object.keys(CATEGORY_Y).find((k) => CATEGORY_Y[k] === v) || ''
                    }
                    stroke="#858585"
                    style={{ fontSize: '10px' }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1e1e1e',
                      border: '1px solid #3e3e42',
                      borderRadius: '4px',
                      fontSize: '11px',
                    }}
                    content={({ payload }) => {
                      const p = payload?.[0]?.payload as TimelineEvent | undefined;
                      if (!p) return null;
                      return (
                        <div className="bg-[#1e1e1e] border border-[#3e3e42] rounded p-2 text-xs max-w-xs">
                          <div className="text-[#4ec9b0]">{p.date}</div>
                          <div className="text-white font-semibold">{p.title}</div>
                          <div className="text-[#858585] mt-1">{p.summary}</div>
                        </div>
                      );
                    }}
                  />
                  <Scatter data={chartData} fill="#007acc">
                    {chartData.map((d, i) => (
                      <Cell key={i} fill={CATEGORY_COLORS[d.category] || '#858585'} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>

              {/* legend */}
              <div className="flex gap-4 mt-2 justify-center text-xs">
                {Object.entries(CATEGORY_COLORS).map(([cat, color]) => (
                  <div key={cat} className="flex items-center gap-1">
                    <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
                    <span className="text-[#858585]">{cat}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* event list */}
            <div className="space-y-2">
              {events.map((e, i) => (
                <div
                  key={i}
                  className="bg-[#252526] border border-[#3e3e42] rounded p-4 flex gap-4"
                >
                  <div
                    className="w-1 rounded"
                    style={{ backgroundColor: CATEGORY_COLORS[e.category] || '#858585' }}
                  />
                  <div className="flex-1">
                    <div className="flex items-baseline gap-3">
                      <span className="text-xs text-[#858585]">{e.date}</span>
                      <span
                        className="text-xs px-2 py-0.5 rounded"
                        style={{
                          backgroundColor: (CATEGORY_COLORS[e.category] || '#858585') + '33',
                          color: CATEGORY_COLORS[e.category] || '#858585',
                        }}
                      >
                        {e.category}
                      </span>
                    </div>
                    <div className="text-sm font-semibold text-white mt-1">{e.title}</div>
                    <div className="text-xs text-[#d4d4d4] mt-1">{e.summary}</div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </RagLayout>
  );
}
