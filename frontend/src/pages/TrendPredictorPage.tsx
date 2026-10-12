import React, { useState, useMemo } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Sparkles,
  Search,
  Coffee,
  Utensils,
  ShoppingBag,
  Scissors,
  Dumbbell,
  Gift,
  Smartphone,
  Store,
  CheckCircle2,
  Clock,
  BarChart2,
  Tag,
  RefreshCw,
  Flame,
  Activity,
  Database,
  Calendar,
  Layers,
  Download,
  Maximize2,
  Copy,
  Check,
  X,
  ChevronDown,
  ChevronUp,
  SlidersHorizontal
} from 'lucide-react';
import {
  fetchSMETrends,
  fetchRelatedKeywords,
  predictTrend
} from '../services/api';
import type {
  SMETrendAnalysisResponse,
  TrendPredictResult,
  KeywordResponse
} from '../types/api';
import { SAMPLE_TREND_PREDICTION, SME_SAMPLE_PREDICTIONS } from '../data/sampleTrendPrediction';
import { LoadingSkeleton } from '../components/common/LoadingSkeleton';
import { ErrorState } from '../components/common/ErrorState';
import { Badge } from '../components/common/Badge';
import { TrendChart } from '../components/charts/TrendChart';

// The 8 official SME Options
export interface SMEOption {
  id: string;
  backendKey: string;
  label: string;
  icon: React.ReactNode;
  defaultKeyword: string;
  tag: string;
}

export const SME_OPTIONS: SMEOption[] = [
  {
    id: 'cafes',
    backendKey: 'cafe',
    label: 'Cafes, Bakeries & Desserts',
    icon: <Coffee size={18} />,
    defaultKeyword: 'matcha latte',
    tag: 'F&B & Beverages',
  },
  {
    id: 'restaurants',
    backendKey: 'restaurant',
    label: 'Restaurants & Cloud Kitchens',
    icon: <Utensils size={18} />,
    defaultKeyword: 'biryani',
    tag: 'Dining & Kitchens',
  },
  {
    id: 'boutiques',
    backendKey: 'boutique',
    label: 'Boutiques & Fashion Stores',
    icon: <ShoppingBag size={18} />,
    defaultKeyword: 'co-ord set',
    tag: 'Apparel & Style',
  },
  {
    id: 'salons',
    backendKey: 'salon',
    label: 'Salons & Beauty Studios',
    icon: <Scissors size={18} />,
    defaultKeyword: 'keratin treatment',
    tag: 'Beauty & Wellness',
  },
  {
    id: 'gyms',
    backendKey: 'gym',
    label: 'Gyms & Yoga Studios',
    icon: <Dumbbell size={18} />,
    defaultKeyword: 'pilates',
    tag: 'Fitness & Health',
  },
  {
    id: 'gifts',
    backendKey: 'gift_shop',
    label: 'Gift & Home Décor Shops',
    icon: <Gift size={18} />,
    defaultKeyword: 'scented candles',
    tag: 'Lifestyle & Gifts',
  },
  {
    id: 'electronics',
    backendKey: 'electronics',
    label: 'Electronics & Accessories',
    icon: <Smartphone size={18} />,
    defaultKeyword: 'wireless earbuds',
    tag: 'Consumer Tech',
  },
  {
    id: 'd2c',
    backendKey: 'd2c',
    label: 'Small D2C & Home Sellers',
    icon: <Store size={18} />,
    defaultKeyword: 'handcrafted pottery',
    tag: 'Artisan & D2C',
  },
];

interface TrendPredictorPageProps {
  onNavigateToCreative?: (promptPreset: string) => void;
}

export const TrendPredictorPage: React.FC<TrendPredictorPageProps> = ({
  onNavigateToCreative,
}) => {
  // Direct Keyword Prediction States
  const [searchKeywords, setSearchKeywords] = useState<string>('artificial intelligence, machine learning');
  const [useTransformer, setUseTransformer] = useState<boolean>(true);
  const [loadingPrediction, setLoadingPrediction] = useState<boolean>(false);
  const [deepForecastResult, setDeepForecastResult] = useState<TrendPredictResult | null>(SAMPLE_TREND_PREDICTION);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  // Active view tab for forecast visualization
  const [activeVisualTab, setActiveVisualTab] = useState<'summary' | 'interactive'>('summary');
  const [forecastDaysFilter, setForecastDaysFilter] = useState<'all' | 'first15' | 'last15'>('all');
  const [copiedSummary, setCopiedSummary] = useState<boolean>(false);
  const [lightboxOpen, setLightboxOpen] = useState<boolean>(false);

  // SME Exploration States
  const [showSmeSection, setShowSmeSection] = useState<boolean>(false);
  const [selectedSme, setSelectedSme] = useState<SMEOption | null>(null);
  const [loadingTrends, setLoadingTrends] = useState<boolean>(false);
  const [trendResult, setTrendResult] = useState<SMETrendAnalysisResponse | null>(null);
  const [trendError, setTrendError] = useState<string | null>(null);

  // Cohere Keyword Pipeline States
  const [cohereResult, setCohereResult] = useState<KeywordResponse | null>(null);
  const [loadingCohere, setLoadingCohere] = useState<boolean>(false);
  const [cohereError, setCohereError] = useState<string | null>(null);
  const [activeForecastingItem, setActiveForecastingItem] = useState<string | null>(null);

  // Parse keyword list from input
  const parsedKeywords = useMemo(() => {
    return searchKeywords
      .split(',')
      .map((k) => k.trim())
      .filter(Boolean);
  }, [searchKeywords]);

  // Execute direct prediction to /predict
  const handleExecutePrediction = async (keywordsToRun?: string[]) => {
    const list = keywordsToRun || parsedKeywords;
    if (list.length === 0) return;

    setLoadingPrediction(true);
    setPredictionError(null);

    try {
      const res = await predictTrend(list, useTransformer);
      setDeepForecastResult(res);
      // Smooth scroll to forecast section
      const el = document.getElementById('trend-forecast-results');
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    } catch (err: unknown) {
      console.error('Trend prediction request error:', err);
      setPredictionError(
        err instanceof Error ? err.message : 'Unable to complete 30-day forecast. Verify backend service is running.'
      );
    } finally {
      setLoadingPrediction(false);
    }
  };

  // Load sample dataset
  const handleLoadSample = () => {
    setSearchKeywords('artificial intelligence, machine learning');
    setDeepForecastResult(SAMPLE_TREND_PREDICTION);
    setPredictionError(null);
    const el = document.getElementById('trend-forecast-results');
    if (el) el.scrollIntoView({ behavior: 'smooth' });
  };

  // Copy forecast summary to clipboard
  const handleCopySummary = () => {
    if (!deepForecastResult) return;
    const text = `
Trend Forecast Report: ${deepForecastResult.topic || deepForecastResult.keywords.join(', ')}
• Current Trend Score: ${deepForecastResult.current_trend_score}
• 15-Day Forecast: ${deepForecastResult.predicted_in_15_days} (${deepForecastResult.change_15d_pct}% ${deepForecastResult.indication_15d})
• 30-Day Forecast: ${deepForecastResult.predicted_in_30_days} (${deepForecastResult.change_30d_pct}% ${deepForecastResult.indication_30d})
• Social Signals Scraped: ${deepForecastResult.rows_collected || deepForecastResult.aggregation_summary?.posts_count || 100}
• Model Source: ${deepForecastResult.model_source || 'database'}
• Observed Window: ${deepForecastResult.aggregation_summary?.start_date || 'N/A'} to ${deepForecastResult.aggregation_summary?.end_date || 'N/A'}
    `.trim();

    navigator.clipboard.writeText(text);
    setCopiedSummary(true);
    setTimeout(() => setCopiedSummary(false), 2000);
  };

  // Handle SME selection
  const handleSelectSME = async (sme: SMEOption) => {
    setSelectedSme(sme);
    setTrendError(null);
    setTrendResult(null);
    setCohereResult(null);
    setCohereError(null);
    setSearchKeywords(sme.defaultKeyword);

    // Immediately load the unique forecast model data for this SME category
    const categorySample = SME_SAMPLE_PREDICTIONS[sme.id] || SME_SAMPLE_PREDICTIONS[sme.backendKey];
    if (categorySample) {
      setDeepForecastResult(categorySample);
    }

    setLoadingTrends(true);

    try {
      const data = await fetchSMETrends(sme.backendKey, {
        city: 'Bengaluru',
        demo: true,
      });
      setTrendResult(data);
    } catch (err: unknown) {
      console.warn('SME Trend analyze error:', err);
      setTrendError(err instanceof Error ? err.message : 'Failed to retrieve trending products.');
    } finally {
      setLoadingTrends(false);
    }
  };

  // Forecast on a specific discovered item
  const handleForecastItem = async (itemName: string) => {
    setActiveForecastingItem(itemName);
    setSearchKeywords(itemName);
    try {
      await handleExecutePrediction([itemName]);
    } finally {
      setActiveForecastingItem(null);
    }
  };

  // Expand with Cohere
  const handleRunCohere = async (seedKeyword: string) => {
    const trimmedSeed = seedKeyword.trim();
    if (!trimmedSeed) return;

    setLoadingCohere(true);
    setCohereError(null);
    setSearchKeywords(trimmedSeed);

    try {
      const smeKey = selectedSme?.backendKey || 'cafe';
      const cohereData = await fetchRelatedKeywords(trimmedSeed, smeKey, 12);
      const relatedKeywords = cohereData.related_keywords?.length ? cohereData.related_keywords : [trimmedSeed];
      setCohereResult({ ...cohereData, related_keywords: relatedKeywords });
      setSearchKeywords(relatedKeywords.join(', '));
      await handleExecutePrediction(relatedKeywords);
    } catch (err: unknown) {
      setCohereError(err instanceof Error ? err.message : 'Failed to expand keywords with Cohere.');
    } finally {
      setLoadingCohere(false);
    }
  };

  // Filtered forecast days
  const displayedForecastDays = useMemo(() => {
    if (!deepForecastResult?.forecast) return [];
    if (forecastDaysFilter === 'first15') {
      return deepForecastResult.forecast.slice(0, 15);
    }
    if (forecastDaysFilter === 'last15') {
      return deepForecastResult.forecast.slice(15);
    }
    return deepForecastResult.forecast;
  }, [deepForecastResult, forecastDaysFilter]);

  const plotImageUrl = useMemo(() => {
    const raw = deepForecastResult?.plot_image ?? '';
    if (typeof raw !== 'string') return '';
    const normalized = raw.trim();
    return normalized.startsWith('data:image/') ? normalized : '';
  }, [deepForecastResult]);

  const hasValidPlotImage = Boolean(plotImageUrl);
  const primaryKeyword = deepForecastResult?.keywords?.[0] || deepForecastResult?.topic || 'trend';
  const plotTitle = primaryKeyword;

  // Combined chart data (Observed + Forecast)
  const combinedChartData = useMemo(() => {
    if (!deepForecastResult) return [];
    const list: Array<{ label: string; value: number; tooltipText: string; isForecast?: boolean }> = [];

    if (deepForecastResult.observed && deepForecastResult.observed.length > 0) {
      deepForecastResult.observed.forEach((pt) => {
        const shortDate = pt.date.slice(5);
        list.push({
          label: shortDate,
          value: pt.observed_trend_score,
          isForecast: false,
          tooltipText: `Observed [${pt.date}]: Score ${pt.observed_trend_score}`,
        });
      });
    }

    if (deepForecastResult.forecast && deepForecastResult.forecast.length > 0) {
      deepForecastResult.forecast.forEach((f) => {
        const shortDate = f.date ? f.date.slice(5) : `D${f.day}`;
        list.push({
          label: shortDate,
          value: f.predicted_score,
          isForecast: true,
          tooltipText: `Forecast Day ${f.day} [${f.date || 'Future'}]: Score ${f.predicted_score}`,
        });
      });
    }

    return list;
  }, [deepForecastResult]);

  return (
    <div className="container" style={{ padding: '2rem 1.25rem 5rem 1.25rem' }}>
      {/* ── Page Header ────────────────────────────────────────────── */}
      <div style={{ maxWidth: '820px', marginBottom: '2.25rem' }}>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.45rem', color: '#818cf8', marginBottom: '0.5rem', fontSize: '0.85rem', fontWeight: 600 }}>
          <TrendingUp size={16} />
          <span>Local Market Intelligence &amp; AI Forecaster</span>
        </div>
        <h1 style={{ fontSize: '2.35rem', fontWeight: 800, marginBottom: '0.75rem', letterSpacing: '-0.025em' }}>
          Trend Predictor
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.98rem', lineHeight: 1.6 }}>
          Harness real-time Reddit and X social listening signals combined with ensemble sentiment analysis (VADER + DistilBERT) to project 30-day velocity trajectories with machine learning.
        </p>
      </div>

      {/* ── SME-first Trend Discovery Console ───────────────────────── */}
      <section className="glass-card" style={{ padding: '1.75rem', marginBottom: '2.5rem', borderColor: 'rgba(99, 102, 241, 0.3)' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: '1rem', marginBottom: '1.25rem' }}>
          <div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Search size={18} color="#818cf8" />
              SME-First Trend Predictor
            </h3>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              Select an SME type, review its trending topics, then use a keyword to expand related terms and forecast the next 30 days.
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <label style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.5rem',
              cursor: 'pointer',
              fontSize: '0.82rem',
              color: useTransformer ? '#a5b4fc' : 'var(--text-muted)',
              backgroundColor: 'rgba(255, 255, 255, 0.04)',
              padding: '0.35rem 0.75rem',
              borderRadius: 'var(--radius-full)',
              border: '1px solid var(--border-subtle)',
            }}>
              <input
                type="checkbox"
                checked={useTransformer}
                onChange={(e) => setUseTransformer(e.target.checked)}
                style={{ accentColor: '#6366f1', cursor: 'pointer' }}
              />
              <span>DistilBERT + VADER Ensemble</span>
            </label>

            <button
              onClick={handleLoadSample}
              className="btn btn-secondary"
              style={{ fontSize: '0.78rem', padding: '0.4rem 0.85rem' }}
              title="Instantly load sample response for Artificial Intelligence & Machine Learning"
            >
              <Sparkles size={14} color="#f59e0b" />
              Load Sample
            </button>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '0.8rem', marginBottom: '1.5rem' }}>
          {SME_OPTIONS.map((option) => {
            const isSelected = selectedSme?.id === option.id;
            return (
              <button
                key={option.id}
                type="button"
                onClick={() => handleSelectSME(option)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: '0.75rem',
                  padding: '0.85rem 0.95rem',
                  borderRadius: 'var(--radius-md)',
                  border: `1px solid ${isSelected ? 'rgba(99, 102, 241, 0.7)' : 'var(--border-subtle)'}`,
                  background: isSelected ? 'rgba(99, 102, 241, 0.14)' : 'rgba(255, 255, 255, 0.02)',
                  color: '#ffffff',
                  textAlign: 'left',
                  cursor: 'pointer',
                  transition: 'all var(--transition-fast)',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  <span style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '2rem',
                    height: '2rem',
                    borderRadius: '50%',
                    backgroundColor: isSelected ? 'rgba(99, 102, 241, 0.28)' : 'rgba(255,255,255,0.05)',
                    color: '#a5b4fc',
                  }}>
                    {option.icon}
                  </span>
                  <div>
                    <div style={{ fontSize: '0.78rem', fontWeight: 700 }}>{option.label}</div>
                    <div style={{ fontSize: '0.66rem', color: 'var(--text-muted)' }}>{option.tag}</div>
                  </div>
                </div>
                {isSelected && <CheckCircle2 size={16} color="#7dd3fc" />}
              </button>
            );
          })}
        </div>

        {selectedSme && (
          <div style={{ display: 'grid', gap: '1rem' }}>
            <div className="glass-card" style={{ padding: '1rem 1.1rem', backgroundColor: 'rgba(15, 23, 42, 0.45)' }}>
              <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: '0.75rem', marginBottom: '0.85rem' }}>
                <div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Selected SME</div>
                  <div style={{ fontSize: '1rem', fontWeight: 700, color: '#ffffff' }}>{selectedSme.label}</div>
                </div>
                <Badge variant="indigo" icon={<Tag size={12} />}>
                  {selectedSme.tag}
                </Badge>
              </div>

              {loadingTrends ? (
                <div style={{ padding: '1rem 0' }}><LoadingSkeleton type="list" count={3} /></div>
              ) : trendError ? (
                <ErrorState title="SME signal error" message={trendError} />
              ) : trendResult && trendResult.trending_items?.length ? (
                <div>
                  <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#e2e8f0', marginBottom: '0.7rem' }}>
                    Trending topics related to {selectedSme.label}
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                    {trendResult.trending_items.slice(0, 8).map((item, idx) => {
                      const itemName = item.name || item.item_name || item.title || item.product || `Topic ${idx + 1}`;
                      return (
                        <button
                          key={`${itemName}-${idx}`}
                          type="button"
                          onClick={() => handleForecastItem(itemName)}
                          style={{
                            padding: '0.45rem 0.75rem',
                            borderRadius: 'var(--radius-full)',
                            border: '1px solid rgba(99, 102, 241, 0.3)',
                            background: activeForecastingItem === itemName ? 'rgba(99, 102, 241, 0.2)' : 'rgba(255,255,255,0.04)',
                            color: '#e2e8f0',
                            cursor: 'pointer',
                            fontSize: '0.74rem',
                          }}
                        >
                          {itemName}
                        </button>
                      );
                    })}
                  </div>
                </div>
              ) : null}
            </div>

            <div className="glass-card" style={{ padding: '1rem 1.1rem', backgroundColor: 'rgba(15, 23, 42, 0.45)' }}>
              <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#e2e8f0', marginBottom: '0.8rem' }}>
                Step 2: Select or enter the keyword
              </div>

              <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                <div style={{ position: 'relative', flex: '1 1 340px' }}>
                  <Search
                    size={18}
                    style={{
                      position: 'absolute',
                      left: '1rem',
                      top: '50%',
                      transform: 'translateY(-50%)',
                      color: 'var(--text-muted)',
                    }}
                  />
                  <input
                    type="text"
                    value={searchKeywords}
                    onChange={(e) => setSearchKeywords(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleRunCohere(searchKeywords);
                    }}
                    placeholder="e.g. matcha latte, seasonal dessert, cafe menu..."
                    className="input"
                    style={{ paddingLeft: '2.75rem', fontSize: '0.92rem' }}
                  />
                </div>

                <button
                  onClick={() => handleRunCohere(searchKeywords)}
                  disabled={loadingCohere || !searchKeywords.trim()}
                  className="btn btn-primary"
                  style={{ padding: '0.7rem 1.3rem', fontSize: '0.9rem' }}
                >
                  {loadingCohere ? (
                    <>
                      <RefreshCw size={16} className="animate-spin" />
                      Expanding Keywords...
                    </>
                  ) : (
                    <>
                      <Sparkles size={16} />
                      Expand &amp; Forecast
                    </>
                  )}
                </button>
              </div>

              {cohereResult && (
                <div style={{ marginTop: '1rem' }}>
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginBottom: '0.55rem' }}>Expanded keyword set</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.45rem' }}>
                    {(cohereResult.related_keywords || []).map((kw, idx) => (
                      <span key={`${kw}-${idx}`} style={{
                        fontSize: '0.72rem',
                        padding: '0.25rem 0.6rem',
                        borderRadius: 'var(--radius-sm)',
                        backgroundColor: 'rgba(99, 102, 241, 0.13)',
                        border: '1px solid rgba(99, 102, 241, 0.25)',
                        color: '#dfe7ff',
                      }}>
                        #{kw}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {!selectedSme && (
          <div style={{ marginTop: '0.8rem', color: 'var(--text-muted)', fontSize: '0.82rem' }}>
            Choose an SME category to unlock trending topic suggestions and the keyword expansion workflow.
          </div>
        )}

        {cohereError && (
          <div style={{ marginTop: '1.1rem' }}>
            <ErrorState title="Keyword expansion failed" message={cohereError} />
          </div>
        )}

        {predictionError && (
          <div style={{ marginTop: '1.25rem' }}>
            <ErrorState
              title="Forecasting Error"
              message={predictionError}
              onRetry={() => handleExecutePrediction()}
            />
          </div>
        )}
      </section>

      {/* ── Active Prediction Loading Banner ────────────────────────── */}
      {loadingPrediction && (
        <div className="glass-card animate-pulse" style={{
          padding: '1.25rem 1.5rem',
          marginBottom: '1.5rem',
          backgroundColor: 'rgba(99, 102, 241, 0.15)',
          border: '1px solid rgba(99, 102, 241, 0.4)',
          borderRadius: 'var(--radius-md)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.85rem',
          color: '#e0e7ff',
        }}>
          <RefreshCw size={20} className="animate-spin" color="#818cf8" />
          <div>
            <div style={{ fontWeight: 700, fontSize: '0.95rem' }}>
              Forecasting 30-Day Trajectory for &ldquo;{searchKeywords}&rdquo;...
            </div>
            <div style={{ fontSize: '0.78rem', color: '#a5b4fc' }}>
              Scraping social velocity signals, computing sentiment ensemble, and running calibrated ML inference.
            </div>
          </div>
        </div>
      )}

      {/* ── Forecast Results Dashboard (When Loaded) ────────────────── */}
      {deepForecastResult && (
        <section id="trend-forecast-results" className="animate-fade-in" style={{ marginBottom: '3.5rem' }}>
          {/* Executive Header Banner */}
          <div className="glass-card" style={{
            padding: '1.75rem 2rem',
            marginBottom: '1.5rem',
            borderColor: 'rgba(99, 102, 241, 0.4)',
            background: 'linear-gradient(135deg, rgba(16, 24, 43, 0.95) 0%, rgba(30, 27, 75, 0.4) 100%)',
          }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1.25rem' }}>
              <div>
                <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.6rem', marginBottom: '0.6rem' }}>
                  <Badge variant="emerald" icon={<CheckCircle2 size={12} />}>
                    Active Forecast Complete
                  </Badge>
                  <span style={{
                    fontSize: '0.75rem',
                    color: '#a5b4fc',
                    backgroundColor: 'rgba(99, 102, 241, 0.15)',
                    padding: '0.2rem 0.6rem',
                    borderRadius: 'var(--radius-full)',
                    border: '1px solid rgba(99, 102, 241, 0.3)',
                    fontFamily: 'var(--font-mono)',
                  }}>
                    Model: {deepForecastResult.model_metadata?.model_name || 'TrendForecaster_v2'} ({deepForecastResult.model_source || 'live_stream'})
                  </span>
                  {deepForecastResult.model_metadata?.metrics && (
                    <span style={{
                      fontSize: '0.75rem',
                      color: '#34d399',
                      backgroundColor: 'rgba(16, 185, 129, 0.1)',
                      padding: '0.2rem 0.6rem',
                      borderRadius: 'var(--radius-full)',
                      border: '1px solid rgba(16, 185, 129, 0.3)',
                      fontFamily: 'var(--font-mono)',
                    }}>
                      MAE: {deepForecastResult.model_metadata.metrics.model_mae} | R²: {deepForecastResult.model_metadata.metrics.model_r2} {(deepForecastResult.model_metadata.metrics.improvement_pct ?? 0) > 0 ? `+${deepForecastResult.model_metadata.metrics.improvement_pct}% vs baseline` : ''}
                    </span>
                  )}
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    • Horizon: 30 Days
                  </span>
                </div>

                <h2 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#ffffff', marginBottom: '0.5rem', letterSpacing: '-0.02em' }}>
                  {primaryKeyword}
                </h2>

                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginTop: '0.5rem' }}>
                  {deepForecastResult.keywords.map((kw, i) => (
                    <span key={i} style={{
                      fontSize: '0.76rem',
                      padding: '0.2rem 0.6rem',
                      borderRadius: 'var(--radius-sm)',
                      backgroundColor: 'rgba(255, 255, 255, 0.05)',
                      border: '1px solid var(--border-subtle)',
                      color: 'var(--text-secondary)',
                    }}>
                      #{kw}
                    </span>
                  ))}
                </div>
              </div>

              {/* Action Buttons */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                <button
                  onClick={handleCopySummary}
                  className="btn btn-secondary"
                  style={{ fontSize: '0.8rem', padding: '0.5rem 0.85rem' }}
                >
                  {copiedSummary ? <Check size={14} color="#10b981" /> : <Copy size={14} />}
                  {copiedSummary ? 'Copied!' : 'Copy Summary'}
                </button>

                {deepForecastResult.plot_image && (
                  <button
                    onClick={() => setLightboxOpen(true)}
                    className="btn btn-secondary"
                    style={{ fontSize: '0.8rem', padding: '0.5rem 0.85rem' }}
                    title="Enlarge Forecast Plot"
                  >
                    <Maximize2 size={14} />
                    Inspect Plot
                  </button>
                )}

                {onNavigateToCreative && (
                  <button
                    onClick={() => {
                      const topic = deepForecastResult.topic || deepForecastResult.keywords.join(' & ');
                      onNavigateToCreative(`Commercial visual campaign highlighting ${topic}, futuristic high tech aesthetic, 8k professional studio lighting`);
                    }}
                    className="btn btn-outline-indigo"
                    style={{ fontSize: '0.8rem', padding: '0.5rem 0.85rem' }}
                  >
                    <Sparkles size={14} color="#c084fc" />
                    Create Ad in Studio
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* ── 4 Primary KPI Metric Cards (Responsive Grid) ─────────── */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            gap: '1rem',
            marginBottom: '1.75rem',
          }}>
            {/* Card 1: Current Score */}
            <div className="glass-card" style={{ padding: '1.35rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  Current Trend Score
                </span>
                <div style={{ width: 32, height: 32, borderRadius: '50%', backgroundColor: 'rgba(99, 102, 241, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#818cf8' }}>
                  <Activity size={16} />
                </div>
              </div>
              <div style={{ fontSize: '2.1rem', fontWeight: 800, color: '#ffffff', fontFamily: 'var(--font-mono)', lineHeight: 1.1, marginBottom: '0.4rem' }}>
                {deepForecastResult.current_trend_score}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <Calendar size={12} />
                <span>Observed baseline average</span>
              </div>
            </div>

            {/* Card 2: 15-Day Milestone */}
            {(() => {
              const chg = deepForecastResult.change_15d_pct ?? deepForecastResult['change_15d_%'] ?? 0;
              const isPositive = chg >= 0;
              return (
                <div className="glass-card" style={{ padding: '1.35rem', borderColor: isPositive ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                    <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                      Day +15 Milestone
                    </span>
                    <Badge variant={isPositive ? 'emerald' : 'rose'} icon={isPositive ? <TrendingUp size={12} /> : <TrendingDown size={12} />}>
                      {isPositive ? '+' : ''}{chg}%
                    </Badge>
                  </div>
                  <div style={{ fontSize: '2.1rem', fontWeight: 800, color: isPositive ? '#6ee7b7' : '#fda4af', fontFamily: 'var(--font-mono)', lineHeight: 1.1, marginBottom: '0.4rem' }}>
                    {deepForecastResult.predicted_in_15_days}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <span>{deepForecastResult.indication_15d}</span>
                    <span style={{ color: 'var(--text-muted)' }}>• Target: {deepForecastResult.forecast?.[14]?.date || 'Day 15'}</span>
                  </div>
                </div>
              );
            })()}

            {/* Card 3: 30-Day Milestone */}
            {(() => {
              const chg = deepForecastResult.change_30d_pct ?? deepForecastResult['change_30d_%'] ?? 0;
              const isPositive = chg >= 0;
              return (
                <div className="glass-card" style={{ padding: '1.35rem', borderColor: isPositive ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                    <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                      Day +30 Horizon
                    </span>
                    <Badge variant={isPositive ? 'emerald' : 'rose'} icon={isPositive ? <TrendingUp size={12} /> : <TrendingDown size={12} />}>
                      {isPositive ? '+' : ''}{chg}%
                    </Badge>
                  </div>
                  <div style={{ fontSize: '2.1rem', fontWeight: 800, color: isPositive ? '#6ee7b7' : '#fda4af', fontFamily: 'var(--font-mono)', lineHeight: 1.1, marginBottom: '0.4rem' }}>
                    {deepForecastResult.predicted_in_30_days}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <span>{deepForecastResult.indication_30d}</span>
                    <span style={{ color: 'var(--text-muted)' }}>• Target: {deepForecastResult.forecast?.[29]?.date || 'Day 30'}</span>
                  </div>
                </div>
              );
            })()}

            {/* Card 4: Social Signals Scraped */}
            <div className="glass-card" style={{ padding: '1.35rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  Data Velocity Sample
                </span>
                <div style={{ width: 32, height: 32, borderRadius: '50%', backgroundColor: 'rgba(6, 182, 212, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#06b6d4' }}>
                  <Layers size={16} />
                </div>
              </div>
              <div style={{ fontSize: '2.1rem', fontWeight: 800, color: '#67e8f9', fontFamily: 'var(--font-mono)', lineHeight: 1.1, marginBottom: '0.4rem' }}>
                {deepForecastResult.rows_collected || deepForecastResult.aggregation_summary?.posts_count || 100}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <Clock size={12} />
                <span>{deepForecastResult.aggregation_summary?.observed_days || 8} Observed Days ({deepForecastResult.aggregation_summary?.continuous_days || 8} continuous)</span>
              </div>
            </div>
          </div>

          {/* ── Aggregation Summary Pipeline Card ────────────────────── */}
          {deepForecastResult.aggregation_summary && (
            <div className="glass-card" style={{
              padding: '1.25rem 1.5rem',
              marginBottom: '1.75rem',
              backgroundColor: 'rgba(99, 102, 241, 0.06)',
              border: '1px solid rgba(99, 102, 241, 0.22)',
            }}>
              <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem' }}>
                  <div style={{
                    width: '2.2rem',
                    height: '2.2rem',
                    borderRadius: 'var(--radius-md)',
                    backgroundColor: 'rgba(99, 102, 241, 0.15)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#818cf8',
                    flexShrink: 0,
                    marginTop: '0.1rem',
                  }}>
                    <Database size={16} />
                  </div>
                  <div>
                    <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.2rem' }}>
                      {deepForecastResult.aggregation_summary.line1 || `Aggregated ${deepForecastResult.aggregation_summary.posts_count} tweets into ${deepForecastResult.aggregation_summary.observed_days} observed days`}
                    </div>
                    <div style={{ fontSize: '0.82rem', color: '#cbd5e1' }}>
                      {deepForecastResult.aggregation_summary.line2 || `Reindexed to ${deepForecastResult.aggregation_summary.continuous_days} continuous days for modeling`}
                    </div>
                  </div>
                </div>

                {/* Timeline metadata badges */}
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', fontSize: '0.76rem' }}>
                  <span style={{ padding: '0.3rem 0.65rem', borderRadius: 'var(--radius-md)', backgroundColor: 'rgba(255, 255, 255, 0.05)', color: 'var(--text-secondary)' }}>
                    Start: <strong style={{ color: '#ffffff' }}>{deepForecastResult.aggregation_summary.start_date}</strong>
                  </span>
                  <span style={{ padding: '0.3rem 0.65rem', borderRadius: 'var(--radius-md)', backgroundColor: 'rgba(255, 255, 255, 0.05)', color: 'var(--text-secondary)' }}>
                    End: <strong style={{ color: '#ffffff' }}>{deepForecastResult.aggregation_summary.end_date}</strong>
                  </span>
                  <span style={{ padding: '0.3rem 0.65rem', borderRadius: 'var(--radius-md)', backgroundColor: 'rgba(99, 102, 241, 0.15)', color: '#a5b4fc', border: '1px solid rgba(99, 102, 241, 0.3)' }}>
                    Posts: <strong style={{ color: '#ffffff' }}>{deepForecastResult.aggregation_summary.posts_count}</strong>
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* ── Visual Analytics Console (Tabs) ──────────────────────── */}
          <div className="glass-card" style={{ padding: '1.75rem', marginBottom: '2rem' }}>
            {/* Tab Navigation Header */}
            <div style={{
              display: 'flex',
              flexWrap: 'wrap',
              justifyContent: 'space-between',
              alignItems: 'center',
              gap: '1rem',
              paddingBottom: '1.25rem',
              borderBottom: '1px solid var(--border-subtle)',
              marginBottom: '1.5rem',
            }}>
              <div>
                <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.2rem' }}>
                  Forecast Trajectory Analytics
                </h3>
                <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                  Switch between the high-res ML plot, interactive SVG trajectory, or the executive table
                </p>
              </div>

              {/* View Switcher Pills */}
              <div style={{
                display: 'inline-flex',
                padding: '0.25rem',
                borderRadius: 'var(--radius-md)',
                backgroundColor: 'rgba(0, 0, 0, 0.3)',
                border: '1px solid var(--border-subtle)',
                gap: '0.25rem',
              }}>
                <button
                  type="button"
                  onClick={() => setActiveVisualTab('summary')}
                  style={{
                    padding: '0.45rem 0.9rem',
                    fontSize: '0.8rem',
                    fontWeight: 600,
                    borderRadius: 'var(--radius-sm)',
                    border: 'none',
                    cursor: 'pointer',
                    backgroundColor: activeVisualTab === 'summary' ? 'var(--accent-primary)' : 'transparent',
                    color: activeVisualTab === 'summary' ? '#ffffff' : 'var(--text-secondary)',
                    transition: 'all var(--transition-fast)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.4rem',
                  }}
                >
                  <Tag size={14} />
                  Summary Table
                </button>

                <button
                  type="button"
                  onClick={() => setActiveVisualTab('interactive')}
                  style={{
                    padding: '0.45rem 0.9rem',
                    fontSize: '0.8rem',
                    fontWeight: 600,
                    borderRadius: 'var(--radius-sm)',
                    border: 'none',
                    cursor: 'pointer',
                    backgroundColor: activeVisualTab === 'interactive' ? 'var(--accent-primary)' : 'transparent',
                    color: activeVisualTab === 'interactive' ? '#ffffff' : 'var(--text-secondary)',
                    transition: 'all var(--transition-fast)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.4rem',
                  }}
                >
                  <Activity size={14} />
                  Interactive Curve
                </button>
              </div>
            </div>

            {/* Tab 1: Summary Table */}
            {activeVisualTab === 'summary' && (
              <div className="animate-fade-in">
                <div className="responsive-table-container">
                  <table className="trend-data-table">
                    <thead>
                      <tr>
                        <th>Topic</th>
                        <th>Current Trend Score</th>
                        <th>Predicted (15d)</th>
                        <th>Change (15d)</th>
                        <th>Indication (15d)</th>
                        <th>Predicted (30d)</th>
                        <th>Change (30d)</th>
                        <th>Indication (30d)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(deepForecastResult.summary_table && deepForecastResult.summary_table.length > 0
                        ? deepForecastResult.summary_table
                        : [
                            {
                              topic: deepForecastResult.topic || deepForecastResult.keywords.join(' OR '),
                              current_trend_score: deepForecastResult.current_trend_score,
                              predicted_in_15_days: deepForecastResult.predicted_in_15_days,
                              'change_15d_%': deepForecastResult['change_15d_%'] ?? deepForecastResult.change_15d_pct,
                              indication_15d: deepForecastResult.indication_15d,
                              predicted_in_30_days: deepForecastResult.predicted_in_30_days,
                              'change_30d_%': deepForecastResult['change_30d_%'] ?? deepForecastResult.change_30d_pct,
                              indication_30d: deepForecastResult.indication_30d,
                            },
                          ]
                      ).map((row, i) => (
                        <tr key={i}>
                          <td style={{ color: '#ffffff', fontWeight: 600, fontFamily: 'var(--font-sans)' }}>
                            {row.topic}
                          </td>
                          <td style={{ color: '#ffffff', fontWeight: 700 }}>
                            {row.current_trend_score}
                          </td>
                          <td style={{ color: '#67e8f9' }}>
                            {row.predicted_in_15_days}
                          </td>
                          <td>
                            <span style={{
                              color: (row['change_15d_%'] ?? row.change_15d_pct ?? 0) >= 0 ? '#10b981' : '#f43f5e',
                              fontWeight: 700,
                            }}>
                              {(row['change_15d_%'] ?? row.change_15d_pct ?? 0) >= 0 ? '+' : ''}
                              {row['change_15d_%'] ?? row.change_15d_pct}%
                            </span>
                          </td>
                          <td>
                            <Badge variant={(row['change_15d_%'] ?? row.change_15d_pct ?? 0) >= 0 ? 'emerald' : 'rose'}>
                              {row.indication_15d}
                            </Badge>
                          </td>
                          <td style={{ color: '#a5b4fc' }}>
                            {row.predicted_in_30_days}
                          </td>
                          <td>
                            <span style={{
                              color: (row['change_30d_%'] ?? row.change_30d_pct ?? 0) >= 0 ? '#10b981' : '#f43f5e',
                              fontWeight: 700,
                            }}>
                              {(row['change_30d_%'] ?? row.change_30d_pct ?? 0) >= 0 ? '+' : ''}
                              {row['change_30d_%'] ?? row.change_30d_pct}%
                            </span>
                          </td>
                          <td>
                            <Badge variant={(row['change_30d_%'] ?? row.change_30d_pct ?? 0) >= 0 ? 'emerald' : 'rose'}>
                              {row.indication_30d}
                            </Badge>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Tab 2: Interactive Dynamic SVG Chart */}
            {activeVisualTab === 'interactive' && (
              <div className="animate-fade-in">
                <div style={{
                  padding: '1.25rem',
                  borderRadius: 'var(--radius-md)',
                  backgroundColor: 'rgba(0, 0, 0, 0.25)',
                  border: '1px solid var(--border-subtle)',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', fontSize: '0.78rem' }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: '#6366f1' }}>
                        <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: '#6366f1' }} />
                        8-Day Observed Baseline
                      </span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: '#06b6d4' }}>
                        <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: '#06b6d4' }} />
                        30-Day Recursive Projection
                      </span>
                    </div>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      Combined Timeline ({combinedChartData.length} Points)
                    </span>
                  </div>

                  <TrendChart
                    title="Historical Observation + 30-Day Trajectory"
                    subtitle="Interactive continuous curve spanning observed social listening dates and model forecast"
                    data={combinedChartData}
                    color="#6366f1"
                    secondaryColor="#f59e0b"
                    primaryLabel="Past Data"
                    secondaryLabel="Forecast Data"
                    showSecondary={true}
                    height={280}
                  />
                </div>
              </div>
            )}

          </div>

          {/* ── 30-Day Daily Forecast Timeline / Table ──────────────────── */}
          {deepForecastResult.forecast && deepForecastResult.forecast.length > 0 && (
            <div className="glass-card" style={{ padding: '1.75rem', marginBottom: '2.5rem' }}>
              <div style={{
                display: 'flex',
                flexWrap: 'wrap',
                justifyContent: 'space-between',
                alignItems: 'center',
                gap: '1rem',
                marginBottom: '1.25rem',
              }}>
                <div>
                  <h4 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                    <Calendar size={16} color="#818cf8" />
                    30-Day Trajectory Daily Breakdown
                  </h4>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    Granular day-by-day score projections output from recursive regressor
                  </p>
                </div>

                {/* Filter buttons */}
                <div style={{ display: 'flex', gap: '0.35rem' }}>
                  <button
                    onClick={() => setForecastDaysFilter('all')}
                    className={`btn ${forecastDaysFilter === 'all' ? 'btn-primary' : 'btn-secondary'}`}
                    style={{ fontSize: '0.75rem', padding: '0.35rem 0.75rem' }}
                  >
                    All 30 Days
                  </button>
                  <button
                    onClick={() => setForecastDaysFilter('first15')}
                    className={`btn ${forecastDaysFilter === 'first15' ? 'btn-primary' : 'btn-secondary'}`}
                    style={{ fontSize: '0.75rem', padding: '0.35rem 0.75rem' }}
                  >
                    Days 1 – 15
                  </button>
                  <button
                    onClick={() => setForecastDaysFilter('last15')}
                    className={`btn ${forecastDaysFilter === 'last15' ? 'btn-primary' : 'btn-secondary'}`}
                    style={{ fontSize: '0.75rem', padding: '0.35rem 0.75rem' }}
                  >
                    Days 16 – 30
                  </button>
                </div>
              </div>

              {/* Scrollable Responsive Table */}
              <div className="responsive-table-container">
                <table className="trend-data-table">
                  <thead>
                    <tr>
                      <th>Day #</th>
                      <th>Target Date</th>
                      <th>Predicted Score</th>
                      <th>Variance vs Baseline</th>
                      <th>Milestone Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {displayedForecastDays.map((item) => {
                      const diff = item.predicted_score - deepForecastResult.current_trend_score;
                      const isMilestone15 = item.day === 15;
                      const isMilestone30 = item.day === 30;

                      return (
                        <tr
                          key={item.day}
                          style={{
                            backgroundColor: isMilestone15 || isMilestone30 ? 'rgba(99, 102, 241, 0.08)' : 'transparent',
                          }}
                        >
                          <td style={{ fontWeight: 700, color: isMilestone15 || isMilestone30 ? '#a5b4fc' : 'var(--text-secondary)' }}>
                            Day {item.day}
                          </td>
                          <td style={{ color: '#ffffff' }}>
                            {item.date}
                          </td>
                          <td style={{ color: '#67e8f9', fontWeight: 700, fontSize: '0.88rem' }}>
                            {item.predicted_score}
                          </td>
                          <td style={{ color: diff >= 0 ? '#10b981' : '#f43f5e' }}>
                            {diff >= 0 ? '+' : ''}{diff.toFixed(2)}
                          </td>
                          <td>
                            {isMilestone15 ? (
                              <Badge variant="amber" icon={<Flame size={12} />}>
                                15-Day Milestone
                              </Badge>
                            ) : isMilestone30 ? (
                              <Badge variant="indigo" icon={<Sparkles size={12} />}>
                                30-Day Milestone
                              </Badge>
                            ) : (
                              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                                Projected Point
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ── Observed Historical Data (8 Days) ─────────────────────── */}
          {deepForecastResult.observed && deepForecastResult.observed.length > 0 && (
            <div className="glass-card" style={{ padding: '1.5rem', marginBottom: '2.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                <div>
                  <h4 style={{ fontSize: '1rem', fontWeight: 700, color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <Activity size={16} color="#06b6d4" />
                    Observed Baseline Period ({deepForecastResult.observed.length} Days)
                  </h4>
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    Continuous daily social sentiment scores extracted prior to ML inference
                  </span>
                </div>
              </div>

              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))',
                gap: '0.75rem',
              }}>
                {deepForecastResult.observed.map((obs, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: '0.75rem',
                      borderRadius: 'var(--radius-sm)',
                      backgroundColor: 'rgba(255, 255, 255, 0.03)',
                      border: '1px solid var(--border-subtle)',
                      textAlign: 'center',
                    }}
                  >
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
                      {obs.date}
                    </span>
                    <span style={{ fontSize: '1.05rem', fontWeight: 700, color: '#ffffff', fontFamily: 'var(--font-mono)' }}>
                      {obs.observed_trend_score}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      )}

      {/* ── Cohere Keyword Expansion Intelligence (If Active) ──────── */}
      {cohereResult && (
        <section className="glass-card animate-fade-in" style={{ padding: '1.5rem', marginBottom: '2.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
            <div>
              <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Tag size={16} color="#818cf8" />
                Cohere LLM Keyword Intelligence
              </h4>
              <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                Seed: <strong style={{ color: '#ffffff' }}>{cohereResult.keyword}</strong> • Generated {cohereResult.total_generated} high-velocity search terms
              </span>
            </div>
            <Badge variant="indigo">Cohere AI</Badge>
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {cohereResult.related_keywords.map((kw, i) => (
              <span
                key={i}
                onClick={() => handleForecastItem(kw)}
                style={{
                  fontSize: '0.78rem',
                  padding: '0.35rem 0.75rem',
                  borderRadius: 'var(--radius-full)',
                  backgroundColor: 'rgba(99, 102, 241, 0.12)',
                  border: '1px solid rgba(99, 102, 241, 0.25)',
                  color: '#a5b4fc',
                  cursor: 'pointer',
                  transition: 'all 150ms ease',
                }}
                title="Run 30-day forecast for this keyword"
              >
                + {kw}
              </span>
            ))}
          </div>
          {cohereError && (
            <div style={{ marginTop: '0.75rem', fontSize: '0.78rem', color: '#f43f5e' }}>
              {cohereError}
            </div>
          )}
        </section>
      )}

      {/* ── SME Industry Verticals Section (Collapsible / Expandable) ─ */}
      <section style={{ marginBottom: '3rem' }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '1.25rem',
          flexWrap: 'wrap',
          gap: '0.75rem',
        }}>
          <div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Store size={18} color="#818cf8" />
              SME Industry Verticals
            </h3>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              Explore pre-indexed categories or scrape real-time market trends for specialized business sectors
            </p>
          </div>

          <button
            onClick={() => setShowSmeSection(!showSmeSection)}
            className="btn btn-secondary"
            style={{ fontSize: '0.8rem', padding: '0.45rem 0.9rem' }}
          >
            <SlidersHorizontal size={14} />
            {showSmeSection ? 'Hide Categories' : 'Browse 8 SME Sectors'}
            {showSmeSection ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
        </div>

        {showSmeSection && (
          <div className="animate-fade-in">
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
              gap: '1rem',
              marginBottom: '2rem',
            }}>
              {SME_OPTIONS.map((sme) => {
                const isSelected = selectedSme?.id === sme.id;
                return (
                  <button
                    key={sme.id}
                    onClick={() => handleSelectSME(sme)}
                    className={`glass-card ${isSelected ? '' : 'glass-card-hover'}`}
                    style={{
                      padding: '1.2rem',
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '0.9rem',
                      textAlign: 'left',
                      cursor: 'pointer',
                      backgroundColor: isSelected ? 'rgba(99, 102, 241, 0.18)' : 'var(--bg-card)',
                      borderColor: isSelected ? 'var(--accent-primary)' : 'var(--border-card)',
                      boxShadow: isSelected ? '0 0 25px rgba(99, 102, 241, 0.3)' : 'none',
                      transition: 'all var(--transition-fast)',
                    }}
                  >
                    <div style={{
                      width: '2.4rem',
                      height: '2.4rem',
                      borderRadius: 'var(--radius-md)',
                      backgroundColor: isSelected ? '#6366f1' : 'rgba(255, 255, 255, 0.05)',
                      color: isSelected ? '#ffffff' : '#a5b4fc',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                    }}>
                      {sme.icon}
                    </div>

                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.2rem', lineHeight: 1.3 }}>
                        {sme.label}
                      </div>
                      <span style={{ fontSize: '0.72rem', color: isSelected ? '#a5b4fc' : 'var(--text-muted)', fontWeight: 500 }}>
                        {sme.tag}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>

            {/* SME Loading State */}
            {loadingTrends && (
              <div style={{ marginBottom: '2rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', color: '#a5b4fc' }}>
                  <Clock size={16} />
                  <span style={{ fontSize: '0.9rem' }}>Analyzing trend signals for {selectedSme?.label}...</span>
                </div>
                <LoadingSkeleton type="card" count={4} />
              </div>
            )}

            {/* SME Error State */}
            {!loadingTrends && trendError && (
              <div style={{ marginBottom: '2rem' }}>
                <ErrorState
                  title="Unable to fetch SME trends"
                  message={trendError}
                  onRetry={() => selectedSme && handleSelectSME(selectedSme)}
                />
              </div>
            )}

            {/* SME Products Grid */}
            {!loadingTrends && !trendError && trendResult?.trending_items && trendResult.trending_items.length > 0 && (
              <div style={{ marginBottom: '2rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '1.25rem' }}>
                  <div>
                    <h4 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#ffffff' }}>
                      Trending Products in {selectedSme?.label}
                    </h4>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      Ranked by engagement velocity and consumer sentiment
                    </p>
                  </div>
                  <Badge variant="emerald" icon={<Flame size={12} />}>
                    {trendResult.trending_items.length} Products Found
                  </Badge>
                </div>

                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
                  gap: '1.25rem',
                }}>
                  {trendResult.trending_items.map((item, index) => {
                    const title = item.name || item.item_name || item.title || item.product || `Item #${index + 1}`;
                    const score = item.trend_score || item.score || Math.round(75 + (index * 5) % 25);
                    const growth = item.growth_rate || item.velocity || '+34%';
                    const desc = item.description || 'Rising consumer interest detected in local social feeds.';

                    return (
                      <div
                        key={index}
                        className="glass-card glass-card-hover"
                        style={{
                          padding: '1.4rem',
                          display: 'flex',
                          flexDirection: 'column',
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                          <span style={{
                            fontSize: '0.7rem',
                            padding: '0.2rem 0.5rem',
                            borderRadius: 'var(--radius-sm)',
                            backgroundColor: 'rgba(99, 102, 241, 0.15)',
                            color: '#a5b4fc',
                            fontFamily: 'var(--font-mono)',
                            fontWeight: 600,
                          }}>
                            #{index + 1} RANK
                          </span>
                          <Badge variant="emerald" icon={<TrendingUp size={12} />}>
                            {String(growth)}
                          </Badge>
                        </div>

                        <h5 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.4rem' }}>
                          {title}
                        </h5>
                        <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '1rem', flex: 1 }}>
                          {desc}
                        </p>

                        <div style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          padding: '0.6rem 0',
                          borderTop: '1px solid var(--border-subtle)',
                          marginBottom: '1rem',
                          fontSize: '0.78rem',
                        }}>
                          <span style={{ color: 'var(--text-muted)' }}>Velocity Score</span>
                          <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#67e8f9' }}>
                            {score}/100
                          </span>
                        </div>

                        <div style={{ display: 'flex', gap: '0.5rem' }}>
                          <button
                            onClick={() => handleForecastItem(title)}
                            disabled={loadingPrediction && activeForecastingItem === title}
                            className="btn btn-primary"
                            style={{ flex: 1, fontSize: '0.78rem', padding: '0.45rem' }}
                          >
                            <BarChart2 size={13} />
                            {loadingPrediction && activeForecastingItem === title ? 'Forecasting...' : '30-Day Forecast'}
                          </button>

                          <button
                            onClick={() => handleRunCohere(title)}
                            disabled={loadingCohere}
                            className="btn btn-secondary"
                            style={{ fontSize: '0.78rem', padding: '0.45rem 0.7rem' }}
                            title="Expand keywords with Cohere"
                          >
                            <Tag size={13} color="#818cf8" />
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </section>

      {/* ── Fullscreen Lightbox Modal for Forecast Plot ──────────────── */}
      {lightboxOpen && hasValidPlotImage && (
        <div
          className="lightbox-overlay"
          onClick={() => setLightboxOpen(false)}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              position: 'relative',
              maxWidth: '92vw',
              maxHeight: '90vh',
              backgroundColor: '#0a0f1d',
              borderRadius: 'var(--radius-lg)',
              border: '1px solid var(--border-card)',
              boxShadow: '0 20px 60px rgba(0, 0, 0, 0.8)',
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            {/* Modal Header */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '1rem 1.5rem',
              borderBottom: '1px solid var(--border-subtle)',
              backgroundColor: 'rgba(15, 23, 42, 0.9)',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <BarChart2 size={16} color="#818cf8" />
                <span style={{ fontSize: '0.95rem', fontWeight: 700, color: '#ffffff' }}>
                  High-Resolution 30-Day Forecast Plot: {plotTitle}
                </span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <a
                  href={plotImageUrl}
                  download={`forecast_plot_${plotTitle.replace(/\s+/g, '_')}.png`}
                  className="btn btn-secondary"
                  style={{ fontSize: '0.78rem', padding: '0.35rem 0.75rem' }}
                >
                  <Download size={13} />
                  Download
                </a>
                <button
                  onClick={() => setLightboxOpen(false)}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: 'var(--text-muted)',
                    cursor: 'pointer',
                    padding: '0.25rem',
                    borderRadius: 'var(--radius-sm)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <X size={20} />
                </button>
              </div>
            </div>

            {/* Image Body */}
            <div style={{ padding: '1rem', overflow: 'auto', display: 'flex', justifyContent: 'center' }}>
              <img
                src={plotImageUrl}
                alt="Enlarged 30-Day Trend Forecast Plot"
                style={{
                  maxWidth: '100%',
                  maxHeight: '78vh',
                  objectFit: 'contain',
                  borderRadius: 'var(--radius-sm)',
                }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
