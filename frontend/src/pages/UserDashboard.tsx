import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { apiFetch, mlFetch } from '../api/client';

interface UserDashboardProps {
  onLogout: () => void;
  tourMode?: boolean;

}

interface PerformanceData {
  batch_number: number;
  accuracy: number;
  correct: number;
  total: number;
}

interface StatsData {
  total_records: number;
  evaluated_records: number;
  keeper_count: number;
  keeper_rate: number;
  discogs_accuracy: number;
  ebay_accuracy: number;
  model_version: string;
  total_batches: number;
  ebay_evaluated?: number;
  ebay_total?: number;
}

interface TodoItem {
  id: string;
  text: string;
  status: 'in-progress' | 'backlog';
}

function UserDashboard({ onLogout }: UserDashboardProps) {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('overview');
  const [performanceData, setPerformanceData] = useState<PerformanceData[]>([]);
  const [stats, setStats] = useState<StatsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [todos, setTodos] = useState<TodoItem[]>([]);
  const [newTodoText, setNewTodoText] = useState('');
  const [draggedItem, setDraggedItem] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState('');

  useEffect(() => {
    loadDashboardData();
    loadTodos();
  }, []);

  const loadTodos = async () => {
    try {
      const response = await apiFetch('/api/todos', {
        credentials: 'include',
      });
      if (response.ok) {
        const data = await response.json();
        setTodos(data);
      }
    } catch (err) {
      console.error('Failed to load todos:', err);
      setTodos([]);
    }
  };

  const handleAddTodo = async (e: React.FormEvent) => {
    e.preventDefault();
    if (newTodoText.trim()) {
      try {
        const response = await apiFetch('/api/todos', {
          method: 'POST',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: newTodoText.trim(),
            status: 'backlog',
            order: todos.length
          })
        });
        if (response.ok) {
          const newTodo = await response.json();
          setTodos([...todos, newTodo]);
          setNewTodoText('');
        }
      } catch (err) {
        console.error('Failed to add todo:', err);
      }
    }
  };

  const handleDragStart = (e: React.DragEvent, itemId: string) => {
    setDraggedItem(itemId);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  };

  const handleDrop = async (e: React.DragEvent, newStatus: 'in-progress' | 'backlog') => {
    e.preventDefault();
    if (draggedItem) {
      try {
        const response = await apiFetch(`/api/todos/${draggedItem}`, {
          method: 'PATCH',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status: newStatus })
        });
        if (response.ok) {
          const updatedTodo = await response.json();
          setTodos(todos.map(todo => 
            todo.id === draggedItem ? updatedTodo : todo
          ));
        }
      } catch (err) {
        console.error('Failed to update todo:', err);
      }
      setDraggedItem(null);
    }
  };

  const handleStartEdit = (todo: TodoItem) => {
    setEditingId(todo.id);
    setEditText(todo.text);
  };

  const handleSaveEdit = async () => {
    if (editingId && editText.trim()) {
      try {
        const response = await apiFetch(`/api/todos/${editingId}`, {
          method: 'PATCH',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: editText.trim() })
        });
        if (response.ok) {
          const updatedTodo = await response.json();
          setTodos(todos.map(todo => 
            todo.id === editingId ? updatedTodo : todo
          ));
        }
      } catch (err) {
        console.error('Failed to update todo:', err);
      }
    }
    setEditingId(null);
    setEditText('');
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditText('');
  };

  const handleDeleteTodo = async (id: string) => {
    try {
      const response = await apiFetch(`/api/todos/${id}`, {
        method: 'DELETE',
        credentials: 'include',
      });
      if (response.ok) {
        setTodos(todos.filter(todo => todo.id !== id));
      }
    } catch (err) {
      console.error('Failed to delete todo:', err);
    }
  };

  const inProgressTodos = todos.filter(t => t.status === 'in-progress');
  const backlogTodos = todos.filter(t => t.status === 'backlog');

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      
      const perfResponse = await mlFetch('/performance_history/', {
        credentials: 'include',
      });
      
      const statsResponse = await mlFetch('/stats/', {
        credentials: 'include',
      });

      const ebayStatsResponse = await mlFetch('/ebay/stats/', {
        credentials: 'include',
      });
      
      if (perfResponse.ok) {
        const perfData = await perfResponse.json();
        const batches = perfData.batches || [];
        const windowSize = 5;
        
        const smoothedData = batches.map((batch: PerformanceData, i: number) => {
          const start = Math.max(0, i - Math.floor(windowSize / 2));
          const end = Math.min(batches.length, start + windowSize);
          const window = batches.slice(start, end);
          const avgAccuracy = window.reduce((sum: number, b: PerformanceData) => sum + b.accuracy, 0) / window.length;

          return {
            batch_number: batch.batch_number,
            accuracy: avgAccuracy * 100,
            correct: batch.correct,
            total: batch.total
          };
        });

        setPerformanceData(smoothedData);
      }
      
      if (statsResponse.ok) {
        const statsData = await statsResponse.json();
        setStats(statsData);
      }

      if (ebayStatsResponse.ok) {
        const ebayData = await ebayStatsResponse.json();
        setStats(prev => prev ? { 
          ...prev, 
          ebay_accuracy: ebayData.ebay_accuracy || 0,
          ebay_evaluated: ebayData.evaluated,
          ebay_total: ebayData.total_listings
        } : null);
      }
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  const sidebarLinks = [
    { id: 'discogs_training', label: 'discogs training', path: '/discogs/training' },
    { id: 'discogs_seller_trigger', label: 'discogs seller trigger', path: '/discogs/seller-trigger' },
    { id: 'discogs_scraper_trigger', label: 'discogs scraper trigger', path: '/discogs/scraper-trigger' },
    { id: 'discogs_inventory_view', label: 'discogs inventory', path: '/discogs/inventory-view' },
    { id: 'trading_platoform_simulator', label: 'trading platform simulator', path: '/trading/simulator' },
    { id: 'wfmu_playlist_parser', label: 'wfmu playlist parser', path: '/wfmu/playlist-parser' },
    { id: 'ebay_auctions', label: 'ebay auctions', path: '/ebay/auctions' },
    { id: 'ebay_buyitnow', label: 'ebay buy it now', path: '/ebay/buyitnow' },
    { id: 'writing', label: 'writing', path: '/writing' },
  ];

  const handleNavigation = (link: typeof sidebarLinks[0]) => {
    if (link.path) {
      navigate(link.path);
    } else {
      setActiveTab(link.id);
    }
  };

  return (
    <div className="min-h-screen bg-[#1e1e1e] text-[#d4d4d4] flex font-mono">
      {/* Nav Sidebar - 20% */}
      <aside className="w-[20%] bg-[#252526] border-r border-[#3e3e42] fixed h-full flex flex-col">
        <nav className="flex-1 p-2 space-y-0.5">
          {sidebarLinks.map((link) => (
            <button
              key={link.id}
              onClick={() => handleNavigation(link)}
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
          <button
            onClick={onLogout}
            className="w-full px-3 py-2 rounded text-xs text-[#cccccc] hover:bg-[#2a2d2e] transition-colors"
          >
            Logout
          </button>
        </div>
      </aside>

      {/* TODO Column - 40% */}
      <div className="ml-[20%] w-[40%] bg-[#1e1e1e] border-r border-[#3e3e42] fixed h-full overflow-hidden p-4 flex flex-col">
        <form onSubmit={handleAddTodo} className="mb-3">
          <input
            type="text"
            value={newTodoText}
            onChange={(e) => setNewTodoText(e.target.value)}
            placeholder="Add new task..."
            className="w-full px-3 py-2 text-xs bg-[#252526] border border-[#3e3e42] rounded text-[#d4d4d4] placeholder-[#6a6a6a] focus:outline-none focus:border-[#007acc]"
          />
        </form>

        <div className="mb-3">
          <div className="text-xs font-semibold text-[#858585] mb-2">IN PROGRESS</div>
          <div 
            className="space-y-1 bg-[#252526] rounded p-2 min-h-[80px]"
            onDragOver={handleDragOver}
            onDrop={(e) => handleDrop(e, 'in-progress')}
          >
            {inProgressTodos.length === 0 ? (
              <div className="text-xs text-[#6a6a6a] p-2 text-center">Drag items here</div>
            ) : (
              inProgressTodos.map(todo => (
                <div
                  key={todo.id}
                  draggable
                  onDragStart={(e) => handleDragStart(e, todo.id)}
                  className="text-xs p-2 bg-[#1e1e1e] rounded border border-[#3e3e42] cursor-move hover:border-[#007acc] flex items-center justify-between group"
                >
                  {editingId === todo.id ? (
                    <input
                      type="text"
                      value={editText}
                      onChange={(e) => setEditText(e.target.value)}
                      onBlur={handleSaveEdit}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleSaveEdit();
                        if (e.key === 'Escape') handleCancelEdit();
                      }}
                      autoFocus
                      className="flex-1 px-2 py-1 text-xs bg-[#252526] border border-[#007acc] rounded text-[#d4d4d4] focus:outline-none"
                    />
                  ) : (
                    <div 
                      className="text-[#d4d4d4] flex-1 cursor-text" 
                      onDoubleClick={() => handleStartEdit(todo)}
                    >
                      {todo.text}
                    </div>
                  )}
                  <button
                    onClick={() => handleDeleteTodo(todo.id)}
                    className="ml-2 text-[#4ec9b0] opacity-0 group-hover:opacity-100 transition-opacity hover:text-[#6ed9c3]"
                  >
                    ✓
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="text-xs font-semibold text-[#858585] mb-2">BACKLOG</div>
          <div 
            className="flex-1 overflow-y-auto space-y-1 bg-[#252526] rounded p-2"
            onDragOver={handleDragOver}
            onDrop={(e) => handleDrop(e, 'backlog')}
          >
            {backlogTodos.map(todo => (
              <div
                key={todo.id}
                draggable
                onDragStart={(e) => handleDragStart(e, todo.id)}
                className="text-xs p-2 bg-[#1e1e1e] rounded border border-[#3e3e42] cursor-move hover:border-[#007acc] flex items-center justify-between group"
              >
                {editingId === todo.id ? (
                  <input
                    type="text"
                    value={editText}
                    onChange={(e) => setEditText(e.target.value)}
                    onBlur={handleSaveEdit}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleSaveEdit();
                      if (e.key === 'Escape') handleCancelEdit();
                    }}
                    autoFocus
                    className="flex-1 px-2 py-1 text-xs bg-[#252526] border border-[#007acc] rounded text-[#d4d4d4] focus:outline-none"
                  />
                ) : (
                  <div 
                    className="text-[#d4d4d4] flex-1 cursor-text" 
                    onDoubleClick={() => handleStartEdit(todo)}
                  >
                    {todo.text}
                  </div>
                )}
                <button
                  onClick={() => handleDeleteTodo(todo.id)}
                  className="ml-2 text-[#4ec9b0] opacity-0 group-hover:opacity-100 transition-opacity hover:text-[#6ed9c3]"
                >
                  ✓
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Performance Dashboard - 40% */}
      <main className="ml-[60%] w-[40%] p-6 overflow-y-auto h-screen">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <div className="animate-spin rounded-full h-12 w-12 border-2 border-[#007acc] border-t-transparent"></div>
          </div>
        ) : (
          <div className="space-y-6">

            <div className="grid grid-cols-3 gap-4">
              <div className="bg-[#252526] border border-[#3e3e42] rounded p-5">
                <p className="text-xs text-[#858585] uppercase tracking-wider mb-2">Total Records</p>
                <p className="text-3xl font-semibold text-white">
                  {stats?.total_records?.toLocaleString() || '0'}
                </p>
                <p className="text-xs text-[#858585] mt-2">
                  {stats?.evaluated_records?.toLocaleString() || '0'} evaluated
                </p>
              </div>

              <div className="bg-[#252526] border border-[#3e3e42] rounded p-5">
                <p className="text-xs text-[#858585] uppercase tracking-wider mb-2">Discogs Accuracy</p>
                <p className={`text-3xl font-semibold ${
                  (stats?.discogs_accuracy || 0) >= 80 ? 'text-[#4ec9b0]' :
                  (stats?.discogs_accuracy || 0) >= 70 ? 'text-[#dcdcaa]' : 'text-[#f48771]'
                }`}>
                  {stats?.discogs_accuracy?.toFixed(1) || '0'}%
                </p>
                <p className="text-xs text-[#858585] mt-2">
                  Last 100 batches
                </p>
              </div>

              <div className="bg-[#252526] border border-[#3e3e42] rounded p-5">
                <p className="text-xs text-[#858585] uppercase tracking-wider mb-2">eBay Accuracy</p>
                {stats?.ebay_accuracy && stats.ebay_accuracy > 0 ? (
                  <>
                    <p className="text-3xl font-semibold text-[#4ec9b0]">
                      {stats.ebay_accuracy.toFixed(1)}%
                    </p>
                    <p className="text-xs text-[#858585] mt-2">
                      {stats.ebay_evaluated} labeled
                    </p>
                  </>
                ) : (
                  <>
                    <p className="text-3xl font-semibold text-[#6a6a6a]">—</p>
                    <p className="text-xs text-[#858585] mt-2">
                      {stats?.ebay_evaluated || 0} labeled, no model yet
                    </p>
                  </>
                )}
              </div>
            </div>

            <div className="space-y-4">
              <div className="bg-[#252526] border border-[#3e3e42] rounded p-5">
                <h3 className="text-sm font-semibold text-white mb-4">Discogs Accuracy Over Time</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={performanceData}>
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
        )}
      </main>
    </div>
  );
}

export default UserDashboard;