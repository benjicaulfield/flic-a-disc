import { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import DiscogsTraining from './pages/discogs/DiscogsTraining';
import DiscogsOOF from './pages/discogs/DiscogsOOF';
import DiscogsKnapsack from './pages/discogs/DiscogsKnapsack';
import DiscogsKnapsackComparison from './pages/discogs/DiscogsKnapsackComparison';
import DiscogsCatalogTraining from './pages/discogs/DiscogsCatalogTraining';
import Deck from './features/deck/Deck';
import TourView from './pages/tour/TourView';
import LandingPage from './pages/landing/LandingPage';
import UserDashboard from './pages/UserDashboard';
import type { User } from "./types";
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
        <Route path="/tour" element={<TourView /> } />
        <Route path="/deck" element={user ? <Deck /> : <Navigate to="/" />} />
        <Route path="/discogs/training" element={user ? <DiscogsTraining /> : <Navigate to="/" />} />
        <Route path="/discogs/oof" element={user ? <DiscogsOOF /> : <Navigate to="/" />} />
        <Route path="/discogs/catalog" element={user ? <DiscogsCatalogTraining /> : <Navigate to="/" />} />
        <Route path="/discogs/knapsack" element={user ? <DiscogsKnapsack /> : <Navigate to="/" />} />
        <Route path="/discogs/knapsack/compare" element={user ? <DiscogsKnapsackComparison /> : <Navigate to="/" />} />
        <Route path="/" />
      </Routes>
    </Router>
  );
}

export default App;
