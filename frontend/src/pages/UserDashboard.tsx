import { useNavigate } from 'react-router-dom';

interface UserDashboardProps {
  onLogout: () => void;
}

function UserDashboard({ onLogout }: UserDashboardProps) {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top Navigation */}
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">flic-a-disc</h1>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => navigate('/')}
                className="text-gray-700 hover:text-gray-900"
              >
                Home
              </button>
              <button
                onClick={(e) => { e.preventDefault(); onLogout(); }}
                className="text-gray-700 hover:text-gray-900"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome Section */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900">Welcome back!</h2>
          <p className="mt-2 text-gray-600">Manage your vinyl collection and discover new records</p>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <button
            onClick={() => navigate('/annotate')}
            className="bg-white p-6 rounded-lg shadow hover:shadow-md transition border border-gray-200 text-left"
          >
            <div className="text-blue-600 text-3xl mb-2">📝</div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Annotate Records</h3>
            <p className="text-gray-600 text-sm">Label eBay listings to improve recommendations</p>
          </button>

          <button
            onClick={() => navigate('/collection')}
            className="bg-white p-6 rounded-lg shadow hover:shadow-md transition border border-gray-200 text-left"
          >
            <div className="text-green-600 text-3xl mb-2">💿</div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">View Collection</h3>
            <p className="text-gray-600 text-sm">Browse your Discogs collection</p>
          </button>

          <button
            onClick={() => navigate('/trading')}
            className="bg-white p-6 rounded-lg shadow hover:shadow-md transition border border-gray-200 text-left"
          >
            <div className="text-purple-600 text-3xl mb-2">🔄</div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Trading Platform</h3>
            <p className="text-gray-600 text-sm">Trade records with other collectors</p>
          </button>
        </div>

        {/* Recent Activity */}
        <div className="bg-white rounded-lg shadow border border-gray-200">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900">Recent Activity</h3>
          </div>
          <div className="px-6 py-8">
            <p className="text-gray-500 text-center">No recent activity</p>
          </div>
        </div>
      </main>
    </div>
  );
}

export default UserDashboard;