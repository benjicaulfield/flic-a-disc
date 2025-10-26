import { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import DiscogsKeepers from './pages/DiscogsKeepers';
import EbayAuctions from './pages/EbayAuctions';
import EbayAnnotation from './pages/EbayAnnotations';
import LandingPage from './pages/landing/LandingPage';
import UserDashboard from './pages/UserDashboard';
import Youbreakifix from './pages/Youbreakifix';
import type { User } from './types/interfaces';


function App() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/auth/me', {
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
  await fetch('http://localhost:8000/api/auth/logout', {
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
        <Route path="/discogs/keepers" element={user ? <DiscogsKeepers /> : <Navigate to="/" />} />
        <Route path="/ebay/auctions" element={user ? <EbayAuctions /> : <Navigate to="/" />} />
        <Route path="/ebay/annotate" element={<EbayAnnotation />} />
        <Route path="/youbreakifix" element={<Youbreakifix />} />
      </Routes>
    </Router>
  );
}

export default App;
