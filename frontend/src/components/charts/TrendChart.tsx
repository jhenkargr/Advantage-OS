import React, { useState } from 'react';

export interface DataPoint {
  label: string | number;
  value: number;
  secondaryValue?: number;
  tooltipText?: string;
  isForecast?: boolean;
}

interface TrendChartProps {
  data: DataPoint[];
  title?: string;
  subtitle?: string;
  height?: number;
  color?: string;
  secondaryColor?: string;
  unit?: string;
  valuePrefix?: string;
  showSecondary?: boolean;
  secondaryLabel?: string;
  primaryLabel?: string;
}

export const TrendChart: React.FC<TrendChartProps> = ({
  data,
  title,
  subtitle,
  height = 240,
  color = '#6366f1',
  secondaryColor = '#06b6d4',
  unit = '',
  valuePrefix = '',
  showSecondary = false,
  primaryLabel = 'Primary',
  secondaryLabel = 'Secondary',
}) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  if (!data || data.length === 0) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
        No chart data available
      </div>
    );
  }

  const padding = { top: 25, right: 30, bottom: 35, left: 45 };
  const width = 640;
  const hasSplitSeries = data.some((d) => typeof d.isForecast === 'boolean');

  const primarySeries = hasSplitSeries ? data.filter((d) => !d.isForecast) : data;
  const secondarySeries = hasSplitSeries
    ? data.filter((d) => d.isForecast)
    : (showSecondary ? data.filter((d) => d.secondaryValue !== undefined) : []);

  const primaryValues = primarySeries.map((d) => d.value);
  const secondaryValues = hasSplitSeries
    ? secondarySeries.map((d) => d.value)
    : data.map((d) => d.secondaryValue ?? 0);
  const allValues = [...primaryValues, ...secondaryValues];

  const minVal = Math.min(0, Math.min(...allValues));
  const maxVal = Math.max(1, Math.max(...allValues) * 1.15);

  const getX = (index: number) => {
    return padding.left + (index / (data.length - 1 || 1)) * (width - padding.left - padding.right);
  };

  const getY = (val: number) => {
    const range = maxVal - minVal || 1;
    return height - padding.bottom - ((val - minVal) / range) * (height - padding.top - padding.bottom);
  };

  const buildPath = (predicate: (d: DataPoint, index: number) => boolean) => {
    const points = data
      .map((d, index) => (predicate(d, index) ? `${getX(index)},${getY(d.value)}` : null))
      .filter((point): point is string => Boolean(point));
    return points.length > 1 ? `M ${points.join(' L ')}` : '';
  };

  const buildAreaPath = (path: string) => {
    if (!path || data.length === 0) return '';
    return `${path} L ${getX(data.length - 1)},${height - padding.bottom} L ${getX(0)},${height - padding.bottom} Z`;
  };

  const primaryPathD = buildPath((d) => hasSplitSeries ? d.isForecast !== true : true);
  const primaryAreaD = buildAreaPath(primaryPathD);

  const lastObservedIndex = hasSplitSeries
    ? data.reduce((last, d, idx) => (!d.isForecast ? idx : last), -1)
    : -1;

  const secondaryPathD = hasSplitSeries
    ? buildPath((d, idx) => d.isForecast === true || idx === lastObservedIndex)
    : (showSecondary ? buildPath((d) => d.secondaryValue !== undefined) : '');

  const legendPrimaryLabel = hasSplitSeries ? 'Past Data' : primaryLabel;
  const legendSecondaryLabel = hasSplitSeries ? 'Forecast Data' : secondaryLabel;
  const shouldShowLegend = showSecondary || hasSplitSeries;

  const gradientId = `trendGrad_${color.replace('#', '')}`;

  return (
    <div style={{ width: '100%' }}>
      {(title || subtitle) && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '1rem' }}>
          <div>
            {title && <h4 style={{ fontSize: '1rem', fontWeight: 700, color: '#ffffff' }}>{title}</h4>}
            {subtitle && <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{subtitle}</p>}
          </div>
          {shouldShowLegend && (
            <div style={{ display: 'flex', gap: '1rem', fontSize: '0.75rem' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: color }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: color }} />
                {legendPrimaryLabel}
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: secondaryColor }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: secondaryColor }} />
                {legendSecondaryLabel}
              </span>
            </div>
          )}
        </div>
      )}

      <div style={{ position: 'relative', width: '100%', overflow: 'hidden' }}>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          style={{ width: '100%', height: 'auto', display: 'block', overflow: 'visible' }}
        >
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity="0.35" />
              <stop offset="100%" stopColor={color} stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {[0, 0.25, 0.5, 0.75, 1].map((pct, idx) => {
            const y = height - padding.bottom - pct * (height - padding.top - padding.bottom);
            const val = minVal + pct * (maxVal - minVal);
            return (
              <g key={idx}>
                <line
                  x1={padding.left}
                  y1={y}
                  x2={width - padding.right}
                  y2={y}
                  stroke="rgba(255, 255, 255, 0.07)"
                  strokeDasharray="4 4"
                />
                <text
                  x={padding.left - 8}
                  y={y + 4}
                  fill="var(--text-muted)"
                  fontSize="10"
                  textAnchor="end"
                  fontFamily="var(--font-mono)"
                >
                  {valuePrefix}{val >= 1000 ? `${(val / 1000).toFixed(1)}k` : Math.round(val)}
                </text>
              </g>
            );
          })}

          {primaryAreaD && <path d={primaryAreaD} fill={`url(#${gradientId})`} />}

          {secondaryPathD && (
            <path
              d={secondaryPathD}
              fill="none"
              stroke={secondaryColor}
              strokeWidth="2"
              strokeDasharray="3 3"
              strokeLinecap="round"
            />
          )}

          {primaryPathD && (
            <path
              d={primaryPathD}
              fill="none"
              stroke={color}
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}

          {data.map((d, i) => {
            const cx = getX(i);
            const cy = getY(d.value);
            const isHovered = hoveredIndex === i;
            const circleColor = d.isForecast ? secondaryColor : color;

            const stride = Math.ceil(data.length / 7);
            const showLabel = i % stride === 0 || i === data.length - 1;

            return (
              <g key={i}>
                {showLabel && (
                  <text
                    x={cx}
                    y={height - 10}
                    fill="var(--text-muted)"
                    fontSize="10"
                    textAnchor="middle"
                    fontFamily="var(--font-mono)"
                  >
                    {String(d.label)}
                  </text>
                )}

                <circle
                  cx={cx}
                  cy={cy}
                  r={isHovered ? 6 : data.length > 25 ? 2.5 : 4}
                  fill={isHovered ? '#ffffff' : circleColor}
                  stroke={isHovered ? circleColor : 'var(--bg-primary)'}
                  strokeWidth="2"
                  style={{ transition: 'all 120ms ease-out', cursor: 'pointer' }}
                  onMouseEnter={() => setHoveredIndex(i)}
                  onMouseLeave={() => setHoveredIndex(null)}
                />
              </g>
            );
          })}
        </svg>

        {hoveredIndex !== null && data[hoveredIndex] && (
          <div
            style={{
              position: 'absolute',
              top: '12px',
              left: `${Math.min(85, Math.max(15, (hoveredIndex / (data.length - 1 || 1)) * 100))}%`,
              transform: 'translateX(-50%)',
              backgroundColor: 'rgba(15, 23, 42, 0.95)',
              border: '1px solid var(--border-hover)',
              borderRadius: 'var(--radius-sm)',
              padding: '0.4rem 0.75rem',
              pointerEvents: 'none',
              boxShadow: '0 4px 14px rgba(0,0,0,0.4)',
              zIndex: 10,
              fontSize: '0.78rem',
              whiteSpace: 'nowrap',
            }}
          >
            <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>
              {data[hoveredIndex].tooltipText || `Index: ${data[hoveredIndex].label}`}
            </div>
            <div style={{ fontWeight: 700, color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: color }} />
              {valuePrefix}{data[hoveredIndex].value.toLocaleString()} {unit}
            </div>
            {showSecondary && data[hoveredIndex].secondaryValue !== undefined && (
              <div style={{ fontWeight: 600, color: secondaryColor, fontSize: '0.72rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: secondaryColor }} />
                {valuePrefix}{data[hoveredIndex].secondaryValue?.toLocaleString()} {unit}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
