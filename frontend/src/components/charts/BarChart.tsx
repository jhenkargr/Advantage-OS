import React, { useState } from 'react';

export interface BarDataPoint {
  label: string;
  value: number;
  benchmark?: number;
  color?: string;
}

interface BarChartProps {
  data: BarDataPoint[];
  title?: string;
  subtitle?: string;
  height?: number;
  valueSuffix?: string;
}

export const BarChart: React.FC<BarChartProps> = ({
  data,
  title,
  subtitle,
  height = 200,
  valueSuffix = '',
}) => {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (!data || data.length === 0) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
        No bar data available
      </div>
    );
  }

  const maxVal = Math.max(1, ...data.map((d) => Math.max(d.value, d.benchmark || 0))) * 1.15;

  return (
    <div style={{ width: '100%' }}>
      {(title || subtitle) && (
        <div style={{ marginBottom: '1.25rem' }}>
          {title && <h4 style={{ fontSize: '1rem', fontWeight: 700, color: '#ffffff' }}>{title}</h4>}
          {subtitle && <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{subtitle}</p>}
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem' }}>
        {data.map((item, idx) => {
          const pct = Math.min(100, Math.max(4, (item.value / maxVal) * 100));
          const benchPct = item.benchmark ? Math.min(100, (item.benchmark / maxVal) * 100) : null;
          const isHovered = hoveredIdx === idx;
          const barColor = item.color || '#6366f1';

          return (
            <div
              key={idx}
              onMouseEnter={() => setHoveredIdx(idx)}
              onMouseLeave={() => setHoveredIdx(null)}
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '0.35rem',
                cursor: 'pointer',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                <span style={{ color: isHovered ? '#ffffff' : 'var(--text-secondary)', fontWeight: 500 }}>
                  {item.label}
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#ffffff' }}>
                  {item.value.toLocaleString()} {valueSuffix}
                </span>
              </div>

              <div style={{
                position: 'relative',
                height: '10px',
                width: '100%',
                backgroundColor: 'rgba(255, 255, 255, 0.06)',
                borderRadius: 'var(--radius-full)',
                overflow: 'hidden',
              }}>
                {/* Benchmark indicator */}
                {benchPct !== null && (
                  <div style={{
                    position: 'absolute',
                    top: 0,
                    bottom: 0,
                    left: `${benchPct}%`,
                    width: '2px',
                    backgroundColor: 'rgba(255, 255, 255, 0.4)',
                    zIndex: 2,
                  }} title={`Benchmark: ${item.benchmark}`} />
                )}

                {/* Primary Bar */}
                <div style={{
                  height: '100%',
                  width: `${pct}%`,
                  borderRadius: 'var(--radius-full)',
                  background: isHovered
                    ? `linear-gradient(90deg, ${barColor}, #ffffff)`
                    : `linear-gradient(90deg, ${barColor}, #818cf8)`,
                  transition: 'width 400ms cubic-bezier(0.16, 1, 0.3, 1), background 150ms ease',
                  boxShadow: isHovered ? `0 0 10px ${barColor}` : 'none',
                }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
