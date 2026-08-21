import { useState, useEffect } from 'react';
import { Login } from '../../components/Login';
import type { LandingPageProps, DiscogsRecord, TodoItem } from '../../types';
import './landing.css';
import { mlFetch, apiFetch } from '../../api/client';



function LandingPage({ onLogin, onLogout }: LandingPageProps) {
  const [recordOfTheDay, setRecordOfTheDay] = useState<DiscogsRecord | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [recentDone, setRecentDone] = useState<TodoItem[]>([]);
  

  useEffect(() => {
    const fetchRecordOfTheDay = async () => {
      try {
        const response = await mlFetch('recommend/rotd/', {
          method: "GET",
        });
        if (!response.ok) {
          throw new Error("failed to fetch documentation");
        }
        const data = await response.json();
        setRecordOfTheDay(data);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setLoading(false);
      }
    };

    

    const fetchRecentDone = async () => {
      try {
        const response = await apiFetch('api/todos/recent');
        if (response.ok) {
          const data = await response.json();
          setRecentDone(data);
        }
      } catch (err) {
        console.error("failed to fetch recent dones", err);
      }
    };

    fetchRecordOfTheDay();
    fetchRecentDone();
  }, []);

  return (
    <div className="landing-container">
      {/* Header */}
      <div className="header-top">
        <div className="header-content">
          <div className="logo">
            <div className="logo-text">
              flic-a-disc.com<br />
            </div>
          </div>
          <div className="logo-description-text">
            leveraging machine learning to help me cross off everything on my wantlist 
          </div>
          <div className="account-links">
            <a href="#" onClick={(e) => { e.preventDefault(); onLogout(); }}>logout</a>
          </div>
        </div>
      </div>

      {/* Navbar */}
      <div className="navbar">
        <a href="/about">about</a> | 
        <a href="/faq">faq</a> | 
        <a href="/writings">writings</a> | 
        <a href="https://https://github.com/benjicaulfield/flic-a-disc/" target="_blank">github</a> | 
        <a href="https://https://www.linkedin.com/in/benjamin-caulfield-265b90159/" target="_blank">linkedIn</a> | 
        <a href="https://bsky.app/profile/benjicaulfield" target="_blank">bsky</a> | 
        <a href="/contact">contact</a> | 
      </div>

      <div className="content">
        {/* Left Column - Record of the Day and Tech Stack */}
        <div className="left-column">
          {/* Record of the Day */}
          <div className="section">
            <div className="section-header">
              RECORD OF THE DAY
            </div>
            <div className="section-content">
              {loading && <p>Loading recommendation...</p>}
              {error && <p>Error: {error}</p>}
              {recordOfTheDay && (
                <div className="record-of-day">
                  <div className='record-of-day-image'>
                    <img 
                      src={recordOfTheDay.record_image || '/placeholder-record.png'}
                      alt={recordOfTheDay.title}
                    />
                  </div>
                  <div className="record-of-day-details">
                    <h2 className="record-title">
                      {recordOfTheDay.artist.toUpperCase()}
                    </h2>
                    <p className="record-subtitle">
                      "{recordOfTheDay.title}", {recordOfTheDay.year}
                    </p>
                    <p className="record-description">
                      {recordOfTheDay.description || "A hidden gem worth exploring from the depths of my wantlist. This record represents everything I love about digging through crates and finding those perfect imperfections that make vinyl collecting an obsession rather than just a hobby."}
                    </p>
                  </div>
                </div>
              )}   
            </div>
          </div>

          {/* Tech Stack - Professional section */}
          <div className="section tech-stack">
            <div className="section-header">
              THIS WEBSITE IS POWERED BY:
            </div>
            <div className="section-content">
              <p><strong>Frontend:</strong> React, TypeScript, Vite, TailwindCSS</p>
              <p><strong>Backend:</strong> Go (Gin), Django REST Framework, PostgreSQL</p>
              <p><strong>ML/AI:</strong> PyTorch Neural Contextual Bandit, scikit-learn TF-IDF, Thompson sampling, online learning</p>
              <p><strong>NLP:</strong> Contrastive encoders, categorical embeddings, custom vectorization</p>
              <p><strong>Pipeline:</strong> Multi-stage filtering (similarity → neural scoring → uncertainty ranking), dynamic thresholds</p>
              <p><strong>Strategy:</strong> Adaptive epsilon-greedy with uncertainty exploration</p>
              <p><strong>Infrastructure:</strong> Custom rate limiter, sparse matrices, batch inference</p>
              <p><strong>Data:</strong> Discogs API, eBay APIs</p>
            </div>
          </div>
        </div>

        {/* Right Column - Login and Statistics */}
        <div className="right-column">
          {/* Login Section */}
          <div className="login-box">
            <div className="login-box-header">Member Login</div>
            <div className="login-form">
              <Login onLogin={onLogin} />
            </div>
          </div>

          
          {/* Recently Completed */}
          <div className="section">
            <div className="section-header">RECENTLY COMPLETED</div>
            <div className="section-content">
              {recentDone.length === 0 ? (
                <p>Nothing completed yet.</p>
              ) : (
                recentDone.map(todo => (
                  <div key={todo.id} style={{ marginBottom: '6px' }}>
                    <span>✓ {todo.text}</span>
                    <span style={{ marginLeft: '8px', opacity: 0.6, fontSize: '0.85em' }}>
                      {new Date(todo.updated_at).toLocaleDateString()}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Tour Section */}
          <div className="section tour-section">
            <div className="section-header">TAKE A TOUR</div>
            <div className="section-content">
              <p style={{ marginBottom: '12px' }}>
                Want to look around? See the neural bandit in action, explore the dashboard,
                watch eBay auctions get scored, and peek at the knapsack solver finding optimal record hauls.
              </p>
              <a href="/tour" style={{ textDecoration: 'none' }}>
                <button
                  style={{
                    width: '100%',
                    padding: '10px',
                    backgroundColor: '#007acc',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '14px',
                    fontWeight: 'bold'
                  }}
                  onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#005a9e'}
                  onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#007acc'}
                >
                  START TOUR →
                </button>
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default LandingPage;
