import { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import DiscogsTraining from './pages/DiscogsTraining';
import DiscogsSellerTrigger from './pages/DiscogsSellerTrigger';
import DiscogsScraperTrigger from './pages/DiscogsScraperTrigger';
import DiscogsInventoryView from './pages/DiscogsInventoryView';
import DiscogsKnapsack from './pages/DiscogsKnapsack';
import DiscogsKeepers from './pages/DiscogsKeepers';
import TradingSimulator from './pages/TradingSimulator';
import WfmuPlaylistParser from './pages/WfmuPlaylistParser';
import EbayAuctions from './pages/EbayAuctions';
import EbayBuyItNow from './pages/EbayBuyItNow';
import EbayKeepers from './pages/EbayKeepers';
import LandingPage from './pages/landing/LandingPage';
import UserDashboard from './pages/UserDashboard';
import type { User } from './types/interfaces';
import { apiFetch } from './api/client';


function App() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = async () => {
    try {
      const response = await apiFetch('api/auth/me', {
        credentials: 'include',
    });

    if (response.ok) {
      const data = await response.json();
      setUser(data);
    }  
  } catch (err) {
    console.log("Not authenticated");
  } finally {
    setLoading(false);
  }
};

const handleLogin = (userData: User) => {
  setUser(userData);
};

const handleLogout = async () => {
  await apiFetch('api/auth/logout', {
    method: 'POST',
    credentials: 'include',
  });
  setUser(null);
};

if (loading) {
  return <div>Loading...</div>
}

return (
    <Router>
      <Routes>
        <Route path="/" element={user 
              ? <Navigate to="/dashboard" replace />
              : <LandingPage onLogin={handleLogin} onLogout={handleLogout} /> } />
        <Route path="/dashboard" element={user
              ? <UserDashboard onLogout={handleLogout} />
              : <Navigate to="/" replace /> } />
        <Route path="/discogs/training" element={user ? <DiscogsTraining /> : <Navigate to="/" />} />
        <Route path="/discogs/knapsack" element={user ? <DiscogsKnapsack /> : <Navigate to="/" />} />
        <Route path="/discogs/seller-trigger" element={user ? <DiscogsSellerTrigger /> : <Navigate to ="/" />} />
        <Route path="/discogs/scraper-trigger" element={user ? <DiscogsScraperTrigger /> : <Navigate to ="/" />} />
        <Route path="/discogs/keepers" element={user ? <DiscogsKeepers /> : <Navigate to ="/" />} />
        <Route path="/discogs/inventory-view" element={user ? <DiscogsInventoryView /> : <Navigate to ="/" />} />
        <Route path="/ebay/auctions" element={user ? <EbayAuctions /> : <Navigate to="/" />} />
        <Route path="/ebay/buyitnow" element={user ? <EbayBuyItNow /> : <Navigate to="/" />} />
        <Route path="/ebay/keepers" element={user ? <EbayKeepers /> : <Navigate to="/" />} />
        <Route path="/wfmu/playlist-parser" element={user ? <WfmuPlaylistParser /> : <Navigate to="/" />} />
        <Route path="/trading/simulator" element={user ? <TradingSimulator /> : <Navigate to="/" />} />


        <Route path="/" />
      </Routes>
    </Router>
  );
}

export default App;
