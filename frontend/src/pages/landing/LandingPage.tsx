import { useState, useEffect } from 'react';
import { Login } from '../../components/Login';
import type { LandingPageProps, DiscogsRecord } from '../../types/interfaces';
import './landing.css';

interface Stats {
  discogs: {
    total: number;
    evaluated: number;
    keepers: number;
    non_keepers: number;
  };
  ebay: {
    total: number;
    evaluated: number;
    enriched: number;
  };
  training: {
    instances: number;
    batch_count: number;
  };
  model: {
    vocab_size: number;
    avg_accuracy: number | null;
  };
}

function LandingPage({ onLogin, onLogout }: LandingPageProps) {
  const [recordOfTheDay, setRecordOfTheDay] = useState<DiscogsRecord | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [statsLoading, setStatsLoading] = useState<boolean>(true);
  const [statsError, setStatsError] = useState<string | null>(null);

  const fetchStats = async () => {
    try {
      const discogsResponse = await fetch('http://localhost:8001/ml/discogs/stats/');
      const ebayResponse = await fetch('http://localhost:8001/ml/ebay/stats/');
      if (!discogsResponse.ok || !ebayResponse.ok) {
        throw new Error(`FAILURE: ${discogsResponse.status} ${ebayResponse.status}`);
      }
      const discogsData = await discogsResponse.json();
      const ebayData = await ebayResponse.json();

      const combinedStats = {
        discogs: {
          total: discogsData.total_records || 0,
          evaluated: discogsData.evaluated_records || 0,
          keepers: discogsData.keeper_count || 0,
          non_keepers: discogsData.non_keepers || 0,
        },
        ebay: {
          total: ebayData.total_listings || 0,
          evaluated: ebayData.evaluated || 0,
          enriched: ebayData.keepers || 0, // Using keepers as enriched count
        },
        training: {
          instances: discogsData.evaluated_records || 0, // Total training instances
          batch_count: discogsData.total_batches || 0,
        },
        model: {
          vocab_size: discogsData.vocab_size || 0,
          avg_accuracy: discogsData.avg_accuracy ? discogsData.avg_accuracy / 100 : 0, // Convert back to decimal
        }
      };
      setStats(combinedStats);
    } catch (err) {
      setStatsError((err as Error).message);
    } finally {
      setStatsLoading(false);
    }
  };

  const fetchRecordOfTheDay = async () => {
    try {
      const response = await fetch('http://localhost:8001/ml/recommend/rotd/');
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
  }
  
  useEffect(() => {
    fetchStats();
    fetchRecordOfTheDay();
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
        <a href="/">home</a> | 
        <a href="/about">about</a> | 
        <a href="/faq">faq</a> | 
        <a href="/writings">writings</a> | 
        <a href="/youbreakifix">you-break-i-fix</a> | 
        <a href="/spotify">softdump spotify</a> | 
        <a href="https://https://github.com/benjicaulfield/flic-a-disc/" target="_blank">github</a> | 
        <a href="https://https://www.linkedin.com/in/benjamin-caulfield-265b90159/" target="_blank">linkedIn</a> | 
        <a href="https://bsky.app/profile/benjicaulfield" target="_blank">bsky</a> | 
        <a href="/contact">contact</a> | 
        <a href="/visitors">visitors</a>
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
              <p><strong>LLM:</strong> Claude API in the manner of music writer Byron Coley</p>
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

          {/* Statistics Section */}
          <div className="section stats-section">
            <div className="section-header">STATISTICS</div>
            <div className="section-content">
              {statsLoading && <p>Loading statistics...</p>}
              {statsError && <p>Error: {statsError}</p>}
              {stats && (
                <div className="stats-grid">
                  <div className="stats-group">
                    <h3>Discogs Collection</h3>
                    <p><strong>Total Records:</strong> {stats.discogs.total.toLocaleString()}</p>
                    <p><strong>Evaluated:</strong> {stats.discogs.evaluated.toLocaleString()}</p>
                    <p><strong>Keepers:</strong> {stats.discogs.keepers.toLocaleString()}</p>
                    <p><strong>Non-Keepers:</strong> {stats.discogs.non_keepers.toLocaleString()}</p>
                  </div>
                  
                  <div className="stats-group">
                    <h3>eBay Listings</h3>
                    <p><strong>Total Listings:</strong> {stats.ebay.total.toLocaleString()}</p>
                    <p><strong>Evaluated:</strong> {stats.ebay.evaluated.toLocaleString()}</p>
                    <p><strong>Enriched:</strong> {stats.ebay.enriched.toLocaleString()}</p>
                  </div>
                  
                  <div className="stats-group">
                    <h3>Machine Learning</h3>
                    <p><strong>Training Examples:</strong> {stats.training.instances.toLocaleString()}</p>
                    <p><strong>Batch Count:</strong> {stats.training.batch_count}</p>
                    <p><strong>Vocab Size:</strong> {stats.model.vocab_size.toLocaleString()}</p>
                    <p><strong>Avg Accuracy:</strong> {stats.model.avg_accuracy ? `${(stats.model.avg_accuracy * 100).toFixed(1)}%` : 'N/A'}</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default LandingPage;
