import React from 'react';

interface LoadingSkeletonProps {
  type?: 'card' | 'table' | 'chart' | 'stats' | 'list';
  count?: number;
}

export const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({
  type = 'card',
  count = 3,
}) => {
  if (type === 'stats') {
    return (
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
        gap: '1rem',
        width: '100%',
      }}>
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="glass-card" style={{ padding: '1.25rem' }}>
            <div className="skeleton" style={{ width: '40%', height: '14px', marginBottom: '1rem' }} />
            <div className="skeleton" style={{ width: '65%', height: '32px', marginBottom: '0.5rem' }} />
            <div className="skeleton" style={{ width: '50%', height: '12px' }} />
          </div>
        ))}
      </div>
    );
  }

  if (type === 'chart') {
    return (
      <div className="glass-card" style={{ padding: '1.5rem', width: '100%' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
          <div className="skeleton" style={{ width: '30%', height: '20px' }} />
          <div className="skeleton" style={{ width: '15%', height: '20px' }} />
        </div>
        <div className="skeleton" style={{ width: '100%', height: '240px', borderRadius: 'var(--radius-md)' }} />
      </div>
    );
  }

  if (type === 'table') {
    return (
      <div className="glass-card" style={{ padding: '1.5rem', width: '100%' }}>
        <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.25rem' }}>
          <div className="skeleton" style={{ width: '25%', height: '36px' }} />
          <div className="skeleton" style={{ width: '20%', height: '36px' }} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {Array.from({ length: count }).map((_, i) => (
            <div key={i} className="skeleton" style={{ width: '100%', height: '48px', borderRadius: 'var(--radius-sm)' }} />
          ))}
        </div>
      </div>
    );
  }

  if (type === 'list') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', width: '100%' }}>
        {Array.from({ length: count }).map((_, i) => (
          <div key={i} className="glass-card" style={{ padding: '1rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div className="skeleton" style={{ width: '42px', height: '42px', borderRadius: '50%' }} />
            <div style={{ flex: 1 }}>
              <div className="skeleton" style={{ width: '35%', height: '16px', marginBottom: '0.4rem' }} />
              <div className="skeleton" style={{ width: '60%', height: '12px' }} />
            </div>
            <div className="skeleton" style={{ width: '80px', height: '28px', borderRadius: 'var(--radius-full)' }} />
          </div>
        ))}
      </div>
    );
  }

  // Default: Grid of Cards
  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
      gap: '1.25rem',
      width: '100%',
    }}>
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="glass-card" style={{ padding: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <div className="skeleton" style={{ width: '50%', height: '18px' }} />
            <div className="skeleton" style={{ width: '25%', height: '22px', borderRadius: 'var(--radius-full)' }} />
          </div>
          <div className="skeleton" style={{ width: '80%', height: '14px', marginBottom: '0.75rem' }} />
          <div className="skeleton" style={{ width: '60%', height: '14px', marginBottom: '1.25rem' }} />
          <div style={{ display: 'flex', gap: '0.5rem', marginTop: 'auto' }}>
            <div className="skeleton" style={{ width: '48%', height: '36px', borderRadius: 'var(--radius-md)' }} />
            <div className="skeleton" style={{ width: '48%', height: '36px', borderRadius: 'var(--radius-md)' }} />
          </div>
        </div>
      ))}
    </div>
  );
};
