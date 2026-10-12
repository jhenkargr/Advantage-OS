import React, { useState, useEffect } from 'react';
import { 
  Users, 
  Search, 
  ExternalLink, 
  ShieldCheck, 
  Flame, 
  Copy, 
  Check, 
  Mail,
  RefreshCw
} from 'lucide-react';
import { fetchInfluencersBySME } from '../services/api';
import type { InfluencerClusterResponse, InfluencerItem } from '../types/api';
import { SME_OPTIONS, type SMEOption } from './TrendPredictorPage';
import { LoadingSkeleton } from '../components/common/LoadingSkeleton';
import { ErrorState } from '../components/common/ErrorState';
import { EmptyState } from '../components/common/EmptyState';
import { Badge } from '../components/common/Badge';
import { Modal } from '../components/common/Modal';

export const InfluencerPage: React.FC = () => {
  const [selectedSme, setSelectedSme] = useState<SMEOption>(SME_OPTIONS[0]);
  const [clusterData, setClusterData] = useState<InfluencerClusterResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [selectedTier, setSelectedTier] = useState<string>('all');
  const [botFreeOnly, setBotFreeOnly] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [sortBy, setSortBy] = useState<'engagement' | 'subscribers' | 'views'>('engagement');

  // Contact / Pitch modal
  const [contactInfluencer, setContactInfluencer] = useState<InfluencerItem | null>(null);
  const [copiedPitch, setCopiedPitch] = useState<boolean>(false);

  // Fetch influencers when SME or bot filter changes
  const loadInfluencers = async (sme: SMEOption, botFree: boolean) => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchInfluencersBySME(sme.backendKey, botFree, 80);
      setClusterData(data);
    } catch (err: unknown) {
      console.warn('Failed loading influencers:', err);
      setError(err instanceof Error ? err.message : 'Failed to retrieve influencer creators.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadInfluencers(selectedSme, botFreeOnly);
  }, [selectedSme, botFreeOnly]);

  const handleSmeChange = (sme: SMEOption) => {
    setSelectedSme(sme);
    setSelectedTier('all');
    setSearchQuery('');
  };

  // Filter & Sort influencers
  const filteredInfluencers = (clusterData?.influencers || []).filter((inf) => {
    // Tier filter
    if (selectedTier !== 'all' && (inf.influencer_tier || '').toLowerCase() !== selectedTier.toLowerCase()) {
      return false;
    }
    // Search query
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const inName = inf.channel_name.toLowerCase().includes(q);
      const inDesc = (inf.description || '').toLowerCase().includes(q);
      const inKw = (inf.keywords || '').toLowerCase().includes(q);
      if (!inName && !inDesc && !inKw) return false;
    }
    return true;
  }).sort((a, b) => {
    if (sortBy === 'engagement') return b.engagement_score - a.engagement_score;
    if (sortBy === 'subscribers') return b.subscriber_count - a.subscriber_count;
    if (sortBy === 'views') return b.avg_views_per_video - a.avg_views_per_video;
    return 0;
  });

  const formatNumber = (num?: number | null): string => {
    if (num == null) return '0';
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}k`;
    return num.toLocaleString();
  };

  const getTierBadgeVariant = (tier?: string) => {
    switch (tier?.toLowerCase()) {
      case 'mega': return 'violet';
      case 'macro': return 'indigo';
      case 'mid': return 'cyan';
      case 'micro': return 'emerald';
      case 'nano': return 'amber';
      default: return 'neutral';
    }
  };

  const generatePitchTemplate = (inf: InfluencerItem) => {
    return `Hi ${inf.channel_name}! We love your local content and would love to partner with you for ${selectedSme.label}. We noticed your audience resonates deeply with ${selectedSme.defaultKeyword}. We'd love to offer an exclusive trial and discuss a paid collaboration. Let us know if you're interested!`;
  };

  const handleCopyPitch = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedPitch(true);
    setTimeout(() => setCopiedPitch(false), 2000);
  };

  return (
    <div className="container" style={{ padding: '2.5rem 1.25rem 5rem 1.25rem' }}>
      {/* Page Title */}
      <div style={{ maxWidth: '780px', marginBottom: '2.5rem' }}>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', color: '#67e8f9', marginBottom: '0.5rem', fontSize: '0.85rem', fontWeight: 600 }}>
          <Users size={16} />
          <span>Local Creator Discovery</span>
        </div>
        <h1 style={{ fontSize: '2.25rem', fontWeight: 800, marginBottom: '0.75rem', letterSpacing: '-0.025em' }}>
          Influencer Matchmaker
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 1.6 }}>
          Select your SME business category below. Our K-Means clustering algorithm maps your sector to verified local creators with authentic follower engagement.
        </p>
      </div>

      {/* SME Type Selector Bar */}
      <section style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#ffffff' }}>
            Target SME Audience
          </h3>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Selected: <strong style={{ color: '#ffffff' }}>{selectedSme.label}</strong>
          </span>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
          gap: '0.75rem',
        }}>
          {SME_OPTIONS.map((sme) => {
            const isSelected = selectedSme.id === sme.id;
            return (
              <button
                key={sme.id}
                onClick={() => handleSmeChange(sme)}
                className={`glass-card ${isSelected ? '' : 'glass-card-hover'}`}
                style={{
                  padding: '0.9rem 1rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  cursor: 'pointer',
                  textAlign: 'left',
                  backgroundColor: isSelected ? 'rgba(6, 182, 212, 0.18)' : 'var(--bg-card)',
                  borderColor: isSelected ? 'var(--accent-cyan)' : 'var(--border-card)',
                  boxShadow: isSelected ? '0 0 20px rgba(6, 182, 212, 0.25)' : 'none',
                }}
              >
                <div style={{
                  color: isSelected ? '#67e8f9' : 'var(--text-muted)',
                  display: 'flex',
                  alignItems: 'center',
                }}>
                  {sme.icon}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{
                    fontSize: '0.86rem',
                    fontWeight: 700,
                    color: '#ffffff',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}>
                    {sme.label}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    {sme.tag}
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </section>

      {/* Cluster Details & Summary Banner */}
      {clusterData && (
        <section style={{ marginBottom: '2rem' }}>
          <div className="glass-card" style={{ padding: '1.5rem', backgroundColor: 'rgba(16, 24, 43, 0.9)' }}>
            <div style={{
              display: 'flex',
              flexWrap: 'wrap',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '1.25rem',
            }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.35rem' }}>
                  <span style={{
                    fontSize: '0.72rem',
                    padding: '0.15rem 0.5rem',
                    borderRadius: 'var(--radius-sm)',
                    backgroundColor: 'rgba(6, 182, 212, 0.15)',
                    color: '#67e8f9',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 700,
                  }}>
                    CLUSTER #{clusterData.cluster_id}
                  </span>
                  <Badge variant="emerald" icon={<ShieldCheck size={12} />}>
                    Verified Niche
                  </Badge>
                </div>
                <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#ffffff', marginBottom: '0.25rem' }}>
                  {clusterData.cluster_label || `Cluster ${clusterData.cluster_id} Creators`}
                </h3>
                <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                  Showing creators active in your geographic and content vertical
                </p>
              </div>

              {/* Quick stats pills */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', fontSize: '0.8rem' }}>
                <div style={{
                  padding: '0.6rem 1rem',
                  borderRadius: 'var(--radius-md)',
                  backgroundColor: 'rgba(255, 255, 255, 0.03)',
                  border: '1px solid var(--border-subtle)',
                }}>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'block' }}>TOTAL POOL</span>
                  <strong style={{ fontSize: '1.1rem', color: '#ffffff', fontFamily: 'var(--font-mono)' }}>
                    {clusterData.count} Creators
                  </strong>
                </div>

                <div style={{
                  padding: '0.6rem 1rem',
                  borderRadius: 'var(--radius-md)',
                  backgroundColor: 'rgba(255, 255, 255, 0.03)',
                  border: '1px solid var(--border-subtle)',
                }}>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'block' }}>FILTERED</span>
                  <strong style={{ fontSize: '1.1rem', color: '#67e8f9', fontFamily: 'var(--font-mono)' }}>
                    {filteredInfluencers.length} Displayed
                  </strong>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Filter & Search Bar */}
      <section style={{ marginBottom: '2rem' }}>
        <div className="glass-card" style={{ padding: '1.25rem 1.5rem' }}>
          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '1rem',
          }}>
            {/* Search Input */}
            <div style={{ position: 'relative', flex: '1 1 240px', minWidth: '220px' }}>
              <Search
                size={16}
                style={{
                  position: 'absolute',
                  left: '0.85rem',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  color: 'var(--text-muted)',
                }}
              />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search creator name, tags, keywords..."
                className="input"
                style={{ paddingLeft: '2.4rem', fontSize: '0.85rem' }}
              />
            </div>

            {/* Tier Filters */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', alignItems: 'center' }}>
              <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginRight: '0.25rem' }}>
                Tier:
              </span>
              {['all', 'nano', 'micro', 'mid', 'macro'].map((tier) => (
                <button
                  key={tier}
                  onClick={() => setSelectedTier(tier)}
                  style={{
                    padding: '0.35rem 0.75rem',
                    borderRadius: 'var(--radius-full)',
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    textTransform: 'capitalize',
                    cursor: 'pointer',
                    backgroundColor: selectedTier === tier ? 'rgba(99, 102, 241, 0.25)' : 'rgba(255, 255, 255, 0.04)',
                    color: selectedTier === tier ? '#ffffff' : 'var(--text-secondary)',
                    border: selectedTier === tier ? '1px solid var(--accent-primary)' : '1px solid var(--border-subtle)',
                    transition: 'all 120ms ease',
                  }}
                >
                  {tier}
                </button>
              ))}
            </div>

            {/* Sort & Bot-Free Toggle */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8rem' }}>
                <span style={{ color: 'var(--text-muted)' }}>Sort by:</span>
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as 'engagement' | 'subscribers' | 'views')}
                  className="select"
                  style={{ padding: '0.4rem 0.75rem', fontSize: '0.8rem', width: 'auto' }}
                >
                  <option value="engagement">Engagement Score</option>
                  <option value="subscribers">Subscribers</option>
                  <option value="views">Avg Views / Video</option>
                </select>
              </div>

              <label style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                cursor: 'pointer',
                fontSize: '0.8rem',
                color: 'var(--text-secondary)',
                userSelect: 'none',
              }}>
                <input
                  type="checkbox"
                  checked={botFreeOnly}
                  onChange={(e) => setBotFreeOnly(e.target.checked)}
                  style={{ accentColor: 'var(--accent-primary)' }}
                />
                Bot-Free Only
              </label>
            </div>
          </div>
        </div>
      </section>

      {/* Loading Skeleton */}
      {loading && (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem', color: '#a5b4fc' }}>
            <RefreshCw size={16} className="animate-spin" />
            <span style={{ fontSize: '0.88rem' }}>Retrieving creator matches for {selectedSme.label}...</span>
          </div>
          <LoadingSkeleton type="card" count={6} />
        </div>
      )}

      {/* Error State */}
      {!loading && error && (
        <ErrorState
          title="Could not fetch influencer creators"
          message={error}
          onRetry={() => loadInfluencers(selectedSme, botFreeOnly)}
        />
      )}

      {/* Empty State */}
      {!loading && !error && filteredInfluencers.length === 0 && (
        <EmptyState
          title="No influencers match your filter"
          description={`We couldn't find any creators matching "${searchQuery || selectedTier}" in this cluster. Try clearing filters or selecting another SME category.`}
          actionText="Clear Filters"
          onAction={() => {
            setSelectedTier('all');
            setSearchQuery('');
          }}
        />
      )}

      {/* Influencers Cards Grid */}
      {!loading && !error && filteredInfluencers.length > 0 && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
          gap: '1.25rem',
        }}>
          {filteredInfluencers.map((inf) => {
            const ytUrl = `https://www.youtube.com/channel/${inf.channel_id}`;
            const tierVariant = getTierBadgeVariant(inf.influencer_tier);

            return (
              <div
                key={inf.channel_id}
                className="glass-card glass-card-hover"
                style={{
                  padding: '1.5rem',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                }}
              >
                <div>
                  {/* Top Header */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                    <Badge variant={tierVariant}>
                      {inf.influencer_tier || 'Creator'}
                    </Badge>
                    <Badge variant="cyan" icon={<Flame size={12} />}>
                      Score: {inf.engagement_score.toFixed(2)}
                    </Badge>
                  </div>

                  {/* Channel Name */}
                  <h4 style={{
                    fontSize: '1.15rem',
                    fontWeight: 700,
                    color: '#ffffff',
                    marginBottom: '0.35rem',
                    lineHeight: 1.3,
                  }}>
                    {inf.channel_name}
                  </h4>

                  {/* Description snippet */}
                  <p style={{
                    fontSize: '0.82rem',
                    color: 'var(--text-secondary)',
                    lineHeight: 1.5,
                    marginBottom: '1.25rem',
                    display: '-webkit-box',
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: 'vertical',
                    overflow: 'hidden',
                    minHeight: '2.4rem',
                  }}>
                    {inf.description || inf.keywords || 'Local video creator producing regular reviews and niche vlogs.'}
                  </p>

                  {/* Creator Key Stats Grid */}
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr 1fr',
                    gap: '0.5rem',
                    padding: '0.85rem',
                    borderRadius: 'var(--radius-sm)',
                    backgroundColor: 'rgba(0, 0, 0, 0.25)',
                    border: '1px solid var(--border-subtle)',
                    marginBottom: '1.25rem',
                    textAlign: 'center',
                  }}>
                    <div>
                      <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', display: 'block' }}>SUBS</span>
                      <strong style={{ fontSize: '0.92rem', color: '#ffffff', fontFamily: 'var(--font-mono)' }}>
                        {formatNumber(inf.subscriber_count)}
                      </strong>
                    </div>
                    <div>
                      <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', display: 'block' }}>VIDEOS</span>
                      <strong style={{ fontSize: '0.92rem', color: '#cbd5e1', fontFamily: 'var(--font-mono)' }}>
                        {formatNumber(inf.video_count)}
                      </strong>
                    </div>
                    <div>
                      <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', display: 'block' }}>AVG VIEWS</span>
                      <strong style={{ fontSize: '0.92rem', color: '#67e8f9', fontFamily: 'var(--font-mono)' }}>
                        {formatNumber(inf.avg_views_per_video)}
                      </strong>
                    </div>
                  </div>
                </div>

                {/* Card Action Buttons */}
                <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
                  <a
                    href={ytUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn btn-secondary"
                    style={{ flex: 1, fontSize: '0.78rem', padding: '0.55rem' }}
                  >
                    <ExternalLink size={14} />
                    View YouTube
                  </a>
                  <button
                    onClick={() => setContactInfluencer(inf)}
                    className="btn btn-primary"
                    style={{ fontSize: '0.78rem', padding: '0.55rem 0.9rem' }}
                  >
                    <Mail size={14} />
                    Pitch
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Pitch Collaboration Dialog Modal */}
      {contactInfluencer && (
        <Modal
          isOpen={!!contactInfluencer}
          onClose={() => setContactInfluencer(null)}
          title={`Collaborate with ${contactInfluencer.channel_name}`}
          subtitle={`Customized partnership inquiry for ${selectedSme.label}`}
        >
          <div>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '1rem',
              borderRadius: 'var(--radius-md)',
              backgroundColor: 'rgba(255, 255, 255, 0.04)',
              border: '1px solid var(--border-subtle)',
              marginBottom: '1.25rem',
            }}>
              <div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Channel Stats</span>
                <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#ffffff' }}>
                  {formatNumber(contactInfluencer.subscriber_count)} Subscribers • {contactInfluencer.engagement_score.toFixed(2)} Engagement Score
                </div>
              </div>
              <Badge variant="cyan">{contactInfluencer.influencer_tier || 'Creator'}</Badge>
            </div>

            <div style={{ marginBottom: '1.25rem' }}>
              <label style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '0.5rem', fontWeight: 600 }}>
                Generated Pitch Message:
              </label>
              <textarea
                rows={5}
                readOnly
                value={generatePitchTemplate(contactInfluencer)}
                className="textarea"
                style={{ fontSize: '0.85rem', lineHeight: 1.5 }}
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
              <button
                onClick={() => setContactInfluencer(null)}
                className="btn btn-secondary"
              >
                Close
              </button>
              <button
                onClick={() => handleCopyPitch(generatePitchTemplate(contactInfluencer))}
                className="btn btn-primary"
              >
                {copiedPitch ? <Check size={16} /> : <Copy size={16} />}
                {copiedPitch ? 'Copied to Clipboard!' : 'Copy Pitch Email'}
              </button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
};
