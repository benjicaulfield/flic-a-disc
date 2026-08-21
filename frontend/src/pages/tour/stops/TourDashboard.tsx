import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

// Mock data - no API calls
const mockPerformanceData = [
  { batch_number: 1, accuracy: 45.2, correct: 226, total: 500 },
  { batch_number: 2, accuracy: 52.8, correct: 264, total: 500 },
  { batch_number: 3, accuracy: 58.4, correct: 292, total: 500 },
  { batch_number: 4, accuracy: 63.1, correct: 315, total: 500 },
  { batch_number: 5, accuracy: 67.9, correct: 339, total: 500 },
  { batch_number: 6, accuracy: 71.2, correct: 356, total: 500 },
  { batch_number: 7, accuracy: 74.8, correct: 374, total: 500 },
  { batch_number: 8, accuracy: 77.3, correct: 386, total: 500 },
  { batch_number: 9, accuracy: 79.6, correct: 398, total: 500 },
  { batch_number: 10, accuracy: 81.4, correct: 407, total: 500 },
];

const mockStats = {
  total_records: 14802,
  evaluated_records: 3240,
  discogs_accuracy: 81.4,
  ebay_accuracy: 0,
  ebay_evaluated: 0,
  ebay_total: 0,
};

const mockTodos = [
  { id: 1, text: 'Fix TF-IDF similarity threshold', status: 'in-progress' },
  { id: 2, text: 'Add Claude API for record descriptions', status: 'in-progress' },
  { id: 3, text: 'Implement batch training scheduler', status: 'backlog' },
  { id: 4, text: 'Add seller inventory pagination', status: 'backlog' },
  { id: 5, text: 'Build knapsack result caching', status: 'backlog' },
  { id: 6, text: 'Add eBay listing refresh cron job', status: 'backlog' },
  { id: 7, text: 'Set up PostgreSQL connection pooling', status: 'done' },
  { id: 8, text: 'Deploy Django ML service to production', status: 'done' },
  { id: 9, text: 'Configure Discogs API rate limiter', status: 'done' },
];

export default function TourDashboard() {
  const [activeTab] = useState('dashboard');
  const [newTodoText] = useState('');
  const [editingId] = useState<number | null>(null);
  const [editText] = useState('');

  const inProgressTodos = mockTodos.filter(t => t.status === 'in-progress');
  const backlogTodos = mockTodos.filter(t => t.status === 'backlog');
  const doneTodos = mockTodos.filter(t => t.status === 'done');

  const sidebarLinks = [
    { id: 'discogs_training', label: 'discogs training', path: '/discogs/training' },
    { id: 'discogs_seller_trigger', label: 'discogs seller trigger', path: '/discogs/seller-trigger' },
    { id: 'discogs_scraper_trigger', label: 'discogs scraper trigger', path: '/discogs/scraper-trigger' },
    { id: 'discogs_inventory_view', label: 'discogs inventory', path: '/discogs/inventory-view' },
    { id: 'discogs_knapsack', label: 'discogs knapsack', path: '/discogs/knapsack' },
    { id: 'trading_platoform_simulator', label: 'trading platform simulator', path: '/trading/simulator' },
    { id: 'wfmu_playlist_parser', label: 'wfmu playlist parser', path: '/wfmu/playlist-parser' },
    { id: 'ebay_auctions', label: 'ebay auctions', path: '/ebay/auctions' },
    { id: 'ebay_buyitnow', label: 'ebay buy it now', path: '/ebay/buyitnow' },
    { id: 'writing', label: 'writing', path: '/writing' },
  ];

  const renderTodoItem = (todo: typeof mockTodos[0]) => (
    <div
      key={todo.id}
      className="text-xs p-2 bg-[#1e1e1e] rounded border border-[#3e3e42] cursor-move hover:border-[#007acc] flex items-center justify-between group"
    >
      {editingId === todo.id ? (
        <input
          type="text"
          value={editText}
          className="flex-1 px-2 py-1 text-xs bg-[#252526] border border-[#007acc] rounded text-[#d4d4d4] focus:outline-none"
        />
      ) : (
        <div className="text-[#d4d4d4] flex-1 cursor-text">
          {todo.text}
        </div>
      )}
      <button className="ml-2 text-[#4ec9b0] opacity-0 group-hover:opacity-100 transition-opacity hover:text-[#6ed9c3]">
        ✓
      </button>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#1e1e1e] text-[#d4d4d4] flex font-mono">
      {/* Nav Sidebar - 20% */}
      <aside className="w-[20%] bg-[#252526] border-r border-[#3e3e42] fixed h-full flex flex-col">
        <nav className="flex-1 p-2 space-y-0.5">
          {sidebarLinks.map((link) => (
            <button
              key={link.id}
              className={`w-full text-left px-3 py-2 rounded text-xs transition-colors ${
                activeTab === link.id && !link.path
                  ? 'bg-[#37373d] text-white'
                  : 'text-[#cccccc] hover:bg-[#2a2d2e]'
              }`}
            >
              {link.label}
            </button>
          ))}
        </nav>

        <div className="p-2 border-t border-[#3e3e42]">
          <button className="w-full px-3 py-2 rounded text-xs text-[#cccccc] hover:bg-[#2a2d2e] transition-colors">
            Logout
          </button>
        </div>
      </aside>

      {/* TODO Column - 40% */}
      <div className="ml-[20%] w-[40%] bg-[#1e1e1e] border-r border-[#3e3e42] fixed h-full overflow-hidden p-4 flex flex-col">
        <form className="mb-3">
          <input
            type="text"
            value={newTodoText}
            placeholder="Add new task..."
            className="w-full px-3 py-2 text-xs bg-[#252526] border border-[#3e3e42] rounded text-[#d4d4d4] placeholder-[#6a6a6a] focus:outline-none focus:border-[#007acc]"
          />
        </form>

        <div className="mb-3">
          <div className="text-xs font-semibold text-[#858585] mb-2">IN PROGRESS</div>
          <div className="space-y-1 bg-[#252526] rounded p-2 min-h-[80px]">
            {inProgressTodos.length === 0 ? (
              <div className="text-xs text-[#6a6a6a] p-2 text-center">Drag items here</div>
            ) : (
              inProgressTodos.map(todo => renderTodoItem(todo))
            )}
          </div>
        </div>

        <div className="mb-3">
          <div className="text-xs font-semibold text-[#858585] mb-2">BACKLOG</div>
          <div className="space-y-1 bg-[#252526] rounded p-2 min-h-[80px]">
            {backlogTodos.length === 0 ? (
              <div className="text-xs text-[#6a6a6a] p-2 text-center">Drag items here</div>
            ) : (
              backlogTodos.map(todo => renderTodoItem(todo))
            )}
          </div>
        </div>

        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="text-xs font-semibold text-[#858585] mb-2">DONE</div>
          <div className="flex-1 overflow-y-auto space-y-1 bg-[#252526] rounded p-2">
            {doneTodos.length === 0 ? (
              <div className="text-xs text-[#6a6a6a] p-2 text-center">Drag items here</div>
            ) : (
              doneTodos.map(todo => renderTodoItem(todo))
            )}
          </div>
        </div>
      </div>

      {/* Performance Dashboard - 40% */}
      <main className="ml-[60%] w-[40%] p-6 overflow-y-auto h-screen">
        <div className="space-y-6">
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-[#252526] border border-[#3e3e42] rounded p-5">
              <p className="text-xs text-[#858585] uppercase tracking-wider mb-2">Total Records</p>
              <p className="text-3xl font-semibold text-white">
                {mockStats.total_records.toLocaleString()}
              </p>
              <p className="text-xs text-[#858585] mt-2">
                {mockStats.evaluated_records.toLocaleString()} evaluated
              </p>
            </div>

            <div className="bg-[#252526] border border-[#3e3e42] rounded p-5">
              <p className="text-xs text-[#858585] uppercase tracking-wider mb-2">Discogs Accuracy</p>
              <p className={`text-3xl font-semibold ${
                mockStats.discogs_accuracy >= 80 ? 'text-[#4ec9b0]' :
                mockStats.discogs_accuracy >= 70 ? 'text-[#dcdcaa]' : 'text-[#f48771]'
              }`}>
                {mockStats.discogs_accuracy.toFixed(1)}%
              </p>
              <p className="text-xs text-[#858585] mt-2">
                Last 100 batches
              </p>
            </div>

            <div className="bg-[#252526] border border-[#3e3e42] rounded p-5">
              <p className="text-xs text-[#858585] uppercase tracking-wider mb-2">eBay Accuracy</p>
              <p className="text-3xl font-semibold text-[#6a6a6a]">—</p>
              <p className="text-xs text-[#858585] mt-2">
                {mockStats.ebay_evaluated} labeled, no model yet
              </p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="bg-[#252526] border border-[#3e3e42] rounded p-5">
              <h3 className="text-sm font-semibold text-white mb-4">Discogs Accuracy Over Time</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={mockPerformanceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#3e3e42" />
                  <XAxis
                    dataKey="batch_number"
                    stroke="#858585"
                    style={{ fontSize: '11px' }}
                    label={{ value: 'Batch Number', position: 'insideBottom', offset: -5, fill: '#858585', fontSize: 11 }}
                  />
                  <YAxis
                    stroke="#858585"
                    style={{ fontSize: '11px' }}
                    label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft', fill: '#858585', fontSize: 11 }}
                    domain={[0, 100]}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1e1e1e',
                      border: '1px solid #3e3e42',
                      borderRadius: '4px',
                      fontSize: '11px'
                    }}
                    labelStyle={{ color: '#d4d4d4' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="accuracy"
                    stroke="#007acc"
                    strokeWidth={2}
                    dot={{ fill: '#007acc', r: 3 }}
                    name="Accuracy"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-[#252526] border border-[#3e3e42] rounded p-5">
              <h3 className="text-sm font-semibold text-white mb-4">eBay Accuracy Over Time</h3>
              <div className="flex items-center justify-center h-[300px]">
                <div className="text-center">
                  <p className="text-[#6a6a6a] text-sm">No eBay training data yet</p>
                  <p className="text-[#4a4a4a] text-xs mt-2">Start annotating eBay listings to see performance</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
