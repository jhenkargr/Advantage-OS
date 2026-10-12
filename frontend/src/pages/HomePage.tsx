import React, { useEffect, useState } from 'react';
import { 
  Sparkles, 
  TrendingUp, 
  Users, 
  Palette, 
  Package, 
  ArrowRight, 
  Database, 
  Cpu, 
  AlertTriangle, 
  CheckCircle2, 
  BarChart3, 
  Zap,
  Activity,
  ChevronRight
} from 'lucide-react';
import type { PageId } from '../components/common/Navbar';
import type { DashboardStats, HealthStatus } from '../types/api';
import { fetchDashboardStats } from '../services/api';
import { LoadingSkeleton } from '../components/common/LoadingSkeleton';
import { ErrorState } from '../components/common/ErrorState';

interface HomePageProps {
  onNavigate: (page: PageId) => void;
  health: HealthStatus | null;
  healthLoading?: boolean;
}

export const HomePage: React.FC<HomePageProps> = ({
  onNavigate,
  health,
}) => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadStats = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchDashboardStats();
      setStats(data);
    } catch (err: unknown) {
      console.warn('Dashboard stats fetch error:', err);
      setError(err instanceof Error ? err.message : 'Could not load executive stats');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStats();
  }, []);

  const featureCards = [
    {
      id: 'trends' as PageId,
      title: 'Trend Predictor',
      badge: 'Core Feature',
      badgeVariant: 'indigo',
      desc: 'Predict rising local demand using Reddit, YouTube & Google Trends with Cohere fallback and 30-day LSTM forecaster.',
      icon: <TrendingUp size={24} color="#818cf8" />,
      color: '#6366f1',
      stats: '8 SME Categories',
    },
    {
      id: 'influencers' as PageId,
      title: 'Influencer Matchmaker',
      badge: 'K-Means Clustered',
      badgeVariant: 'cyan',
      desc: 'Find verified, bot-free local YouTube creators matched to your exact niche with engagement rankings and audience reach.',
      icon: <Users size={24} color="#67e8f9" />,
      color: '#06b6d4',
      stats: '1,500+ Local Creators',
    },
    {
      id: 'creative' as PageId,
      title: 'Creative Studio',
      badge: 'NVIDIA + Groq',
      badgeVariant: 'violet',
      desc: 'Generate commercial product visuals with Text-to-Image (Flux), transform existing product photos, and draft high-converting ad captions.',
      icon: <Palette size={24} color="#c084fc" />,
      color: '#8b5cf6',
      stats: '3 Creative Pipelines',
    },
    {
      id: 'inventory' as PageId,
      title: 'Inventory & Demand',
      badge: 'ML Forecasting',
      badgeVariant: 'emerald',
      desc: 'Track 80+ products with intermittent demand forecasting, stockout risk alerts, and automated stock replenishment plans.',
      icon: <Package size={24} color="#6ee7b7" />,
      color: '#10b981',
      stats: 'TSB + Poisson Blended',
    },
  ];

  return (
    <div className="container" style={{ padding: '2.5rem 1.25rem 4rem 1.25rem' }}>
      {/* Hero Section */}
      <section style={{
        textAlign: 'center',
        maxWidth: '840px',
        margin: '0 auto 3.5rem auto',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}>
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.5rem',
          padding: '0.35rem 0.9rem',
          borderRadius: 'var(--radius-full)',
          background: 'rgba(99, 102, 241, 0.12)',
          border: '1px solid rgba(99, 102, 241, 0.3)',
          color: '#a5b4fc',
          fontSize: '0.8rem',
          fontWeight: 600,
          marginBottom: '1.25rem',
          boxShadow: '0 0 20px rgba(99, 102, 241, 0.2)',
        }}>
          <Sparkles size={14} />
          <span>Unified SME Growth Operating System</span>
        </div>

        <h1 style={{
          fontSize: 'clamp(2.2rem, 5vw, 3.4rem)',
          fontWeight: 800,
          lineHeight: 1.15,
          marginBottom: '1.25rem',
          letterSpacing: '-0.03em',
        }}>
          Empowering Local SMEs with <br />
          <span className="gradient-text-accent">AI Trends, Influencers & Demand</span>
        </h1>

        <p style={{
          fontSize: 'clamp(1rem, 2vw, 1.15rem)',
          color: 'var(--text-secondary)',
          lineHeight: 1.6,
          marginBottom: '2rem',
          maxWidth: '680px',
        }}>
          Discover upcoming local consumer trends, match with high-converting creators, generate marketing visuals on demand, and optimize stock replenishment with ML forecasting.
        </p>

        {/* Hero CTA buttons */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', justifyContent: 'center' }}>
          <button
            onClick={() => onNavigate('trends')}
            className="btn btn-primary"
            style={{ padding: '0.8rem 1.75rem', fontSize: '0.95rem' }}
          >
            <TrendingUp size={18} />
            Explore Trend Predictor
            <ArrowRight size={16} />
          </button>
          <button
            onClick={() => onNavigate('inventory')}
            className="btn btn-secondary"
            style={{ padding: '0.8rem 1.5rem', fontSize: '0.95rem' }}
          >
            <Package size={18} />
            View Product Inventory
          </button>
        </div>
      </section>

      {/* Live System Health & Backend Status Banner */}
      <section style={{ marginBottom: '3rem' }}>
        <div className="glass-card" style={{ padding: '1.25rem 1.75rem' }}>
          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '1rem',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div style={{
                width: '2.5rem',
                height: '2.5rem',
                borderRadius: 'var(--radius-md)',
                backgroundColor: 'rgba(16, 185, 129, 0.12)',
                border: '1px solid rgba(16, 185, 129, 0.25)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#10b981',
              }}>
                <Activity size={20} />
              </div>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#ffffff' }}>
                    AdVantage OS Gateway Status
                  </h4>
                  <span className={`status-beacon ${health?.database_ready ? 'online' : 'warning'}`} />
                </div>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Connected to Neon Serverless PostgreSQL &amp; Local FastAPI Orchestrator
                </p>
              </div>
            </div>

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.25rem', fontSize: '0.8rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: health?.database_ready ? '#6ee7b7' : 'var(--text-muted)' }}>
                <Database size={15} />
                <span>Database: {health?.database_ready ? 'Neon Connected' : 'Checking...'}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: health?.models_ready ? '#a5b4fc' : 'var(--text-muted)' }}>
                <Cpu size={15} />
                <span>Forecasting: {health?.models_ready ? 'Models Active' : 'Checking...'}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: health?.creative_studio_ready ? '#c084fc' : 'var(--text-muted)' }}>
                <Zap size={15} />
                <span>Creative Studio: {health?.creative_studio_ready ? 'NVIDIA / Groq Ready' : 'Configured'}</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* KPI Overview Metrics */}
      <section style={{ marginBottom: '3.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '1.25rem' }}>
          <div>
            <h2 style={{ fontSize: '1.35rem', fontWeight: 700, color: '#ffffff' }}>
              Consolidated Business Metrics
            </h2>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              Real-time telemetry gathered across inventory logs, alerts, and market signals
            </p>
          </div>
          <button 
            onClick={loadStats} 
            className="btn btn-secondary" 
            style={{ fontSize: '0.78rem', padding: '0.4rem 0.8rem' }}
          >
            Refresh Metrics
          </button>
        </div>

        {loading ? (
          <LoadingSkeleton type="stats" />
        ) : error ? (
          <ErrorState message={error} onRetry={loadStats} />
        ) : stats ? (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: '1.25rem',
          }}>
            <div className="glass-card glass-card-hover" style={{ padding: '1.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)', fontWeight: 600 }}>TOTAL CATALOG SKUs</span>
                <Package size={18} color="#818cf8" />
              </div>
              <div style={{ fontSize: '2rem', fontWeight: 800, color: '#ffffff', fontFamily: 'var(--font-mono)' }}>
                {stats.total_skus}
              </div>
              <div style={{ fontSize: '0.78rem', color: '#10b981', marginTop: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                <CheckCircle2 size={13} /> Active SME Inventory Catalog
              </div>
            </div>

            <div className="glass-card glass-card-hover" style={{ padding: '1.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)', fontWeight: 600 }}>STOCK UNITS MONITORED</span>
                <BarChart3 size={18} color="#67e8f9" />
              </div>
              <div style={{ fontSize: '2rem', fontWeight: 800, color: '#ffffff', fontFamily: 'var(--font-mono)' }}>
                {stats.total_inventory_units.toLocaleString()}
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                Current live warehouse stock
              </div>
            </div>

            <div className="glass-card glass-card-hover" style={{ padding: '1.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)', fontWeight: 600 }}>STOCKOUT RISK ALERTS</span>
                <AlertTriangle size={18} color={stats.critical_stockout_alerts > 0 ? '#f43f5e' : '#10b981'} />
              </div>
              <div style={{ 
                fontSize: '2rem', 
                fontWeight: 800, 
                color: stats.critical_stockout_alerts > 0 ? '#f43f5e' : '#10b981',
                fontFamily: 'var(--font-mono)' 
              }}>
                {stats.critical_stockout_alerts}
              </div>
              <div style={{ 
                fontSize: '0.78rem', 
                color: stats.critical_stockout_alerts > 0 ? '#fda4af' : '#6ee7b7', 
                marginTop: '0.5rem' 
              }}>
                {stats.critical_stockout_alerts > 0 ? 'Requires immediate reorder' : 'All SKUs within safe levels'}
              </div>
            </div>

            <div className="glass-card glass-card-hover" style={{ padding: '1.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)', fontWeight: 600 }}>CREATOR POOL</span>
                <Users size={18} color="#c084fc" />
              </div>
              <div style={{ fontSize: '2rem', fontWeight: 800, color: '#ffffff', fontFamily: 'var(--font-mono)' }}>
                1,566
              </div>
              <div style={{ fontSize: '0.78rem', color: '#c084fc', marginTop: '0.5rem' }}>
                Clustered YouTube channels
              </div>
            </div>
          </div>
        ) : null}
      </section>

      {/* Feature Navigation Cards */}
      <section style={{ marginBottom: '4rem' }}>
        <div style={{ marginBottom: '1.5rem' }}>
          <h2 style={{ fontSize: '1.35rem', fontWeight: 700, color: '#ffffff' }}>
            Application Modules
          </h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Direct access to all 4 intelligence workflows
          </p>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '1.5rem',
        }}>
          {featureCards.map((feat) => (
            <div
              key={feat.id}
              className="glass-card glass-card-hover"
              onClick={() => onNavigate(feat.id)}
              style={{
                padding: '1.75rem',
                display: 'flex',
                flexDirection: 'column',
                cursor: 'pointer',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.25rem' }}>
                <div style={{
                  width: '3rem',
                  height: '3rem',
                  borderRadius: 'var(--radius-md)',
                  backgroundColor: 'rgba(255, 255, 255, 0.05)',
                  border: `1px solid ${feat.color}40`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: `0 0 15px ${feat.color}25`,
                }}>
                  {feat.icon}
                </div>
                <span style={{
                  fontSize: '0.72rem',
                  fontWeight: 700,
                  padding: '0.2rem 0.6rem',
                  borderRadius: 'var(--radius-full)',
                  backgroundColor: `${feat.color}20`,
                  color: feat.color,
                  border: `1px solid ${feat.color}40`,
                }}>
                  {feat.badge}
                </span>
              </div>

              <h3 style={{ fontSize: '1.2rem', marginBottom: '0.6rem', color: '#ffffff' }}>
                {feat.title}
              </h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.55, marginBottom: '1.5rem', flex: 1 }}>
                {feat.desc}
              </p>

              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                paddingTop: '1rem',
                borderTop: '1px solid var(--border-subtle)',
                fontSize: '0.82rem',
              }}>
                <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  {feat.stats}
                </span>
                <span style={{ color: feat.color, fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                  Launch <ChevronRight size={15} />
                </span>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* SME Architecture Workflow Guide */}
      <section className="glass-card" style={{ padding: '2.5rem' }}>
        <div style={{ textAlign: 'center', maxWidth: '640px', margin: '0 auto 2.5rem auto' }}>
          <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem', color: '#ffffff' }}>
            The SME Closed-Loop Growth Workflow
          </h2>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
            How AdVantage OS turns social noise into tangible sales and automated operations
          </p>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: '1.5rem',
          position: 'relative',
        }}>
          {[
            {
              step: '01',
              title: 'Detect Rising Trends',
              desc: 'Continuous social listening across Reddit & YouTube clusters local demand signals.',
            },
            {
              step: '02',
              title: 'Match Creators',
              desc: 'Connect with verified niche creators to amplify your offer to local shoppers.',
            },
            {
              step: '03',
              title: 'Produce Creative Assets',
              desc: 'Generate studio-grade product graphics and compelling ad copy in seconds.',
            },
            {
              step: '04',
              title: 'Forecast & Replenish',
              desc: 'ML demand models predict sales velocity and generate stock reorder schedules.',
            },
          ].map((s) => (
            <div key={s.step} style={{
              padding: '1.25rem',
              borderRadius: 'var(--radius-md)',
              backgroundColor: 'rgba(255, 255, 255, 0.02)',
              border: '1px solid var(--border-subtle)',
            }}>
              <span style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '1.2rem',
                fontWeight: 800,
                color: 'var(--accent-primary)',
                display: 'block',
                marginBottom: '0.5rem',
              }}>
                {s.step}
              </span>
              <h4 style={{ fontSize: '1rem', color: '#ffffff', marginBottom: '0.4rem' }}>
                {s.title}
              </h4>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                {s.desc}
              </p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};
