import React from 'react';
import { Sparkles, Terminal, Database, Cpu, ShieldCheck } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer style={{
      marginTop: 'auto',
      borderTop: '1px solid var(--border-subtle)',
      backgroundColor: 'rgba(7, 10, 18, 0.95)',
      padding: '2.5rem 0 2rem 0',
      fontSize: '0.85rem',
      color: 'var(--text-muted)',
    }}>
      <div className="container">
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: '2rem',
          marginBottom: '2rem',
        }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#ffffff', marginBottom: '0.75rem' }}>
              <Sparkles size={18} color="#818cf8" />
              <span style={{ fontWeight: 700, fontSize: '1rem' }}>AdVantage OS</span>
            </div>
            <p style={{ fontSize: '0.82rem', lineHeight: 1.6, color: 'var(--text-muted)' }}>
              Next-generation intelligence and creative automation platform engineered specifically for local businesses and SMEs.
            </p>
          </div>

          <div>
            <h4 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.75rem' }}>
              Architecture
            </h4>
            <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.4rem', fontSize: '0.8rem' }}>
              <li style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Database size={13} color="#6ee7b7" /> Neon Serverless PostgreSQL
              </li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Cpu size={13} color="#a5b4fc" /> DistilBERT + VADER ML Models
              </li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Terminal size={13} color="#67e8f9" /> Cohere Command & Embeddings
              </li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Sparkles size={13} color="#c084fc" /> NVIDIA API Catalog & Groq
              </li>
            </ul>
          </div>

          <div>
            <h4 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.75rem' }}>
              Core Capabilities
            </h4>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
              {['Reddit & X Scraping', 'Topic Modeling', '30D LSTM Forecast', 'Micro Influencers', 'Text2Img Flux', 'Stock Plan TSB'].map((t) => (
                <span key={t} style={{
                  fontSize: '0.72rem',
                  padding: '0.2rem 0.5rem',
                  borderRadius: 'var(--radius-sm)',
                  background: 'rgba(255, 255, 255, 0.04)',
                  border: '1px solid var(--border-subtle)',
                  color: 'var(--text-secondary)',
                }}>
                  {t}
                </span>
              ))}
            </div>
          </div>
        </div>

        <div style={{
          paddingTop: '1.5rem',
          borderTop: '1px solid var(--border-subtle)',
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '1rem',
          fontSize: '0.78rem',
        }}>
          <div>
            &copy; {new Date().getFullYear()} AdVantage OS. Built for Small & Medium Enterprises.
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#10b981' }}>
            <ShieldCheck size={14} /> Production Ready • REST API Gateway Connected
          </div>
        </div>
      </div>
    </footer>
  );
};
