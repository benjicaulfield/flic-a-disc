import { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import DiscogsKeepers from './pages/DiscogsKeepers';
import EbayAuctions from './pages/EbayAuctions';
import EbayAnnotation from './pages/EbayAnnotations';
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
        <Route path="/training/discogs" element={user ? <DiscogsKeepers /> : <Navigate to="/" />} />
        <Route path="/ebay/auctions" element={user ? <EbayAuctions /> : <Navigate to="/" />} />
        <Route path="/ebay/annotate" element={<EbayAnnotation />} />
      </Routes>
    </Router>
  );
}

export default App;
