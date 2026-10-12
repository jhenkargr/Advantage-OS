import React from 'react';
import { Inbox } from 'lucide-react';

interface EmptyStateProps {
  title?: string;
  description?: string;
  icon?: React.ReactNode;
  actionText?: string;
  onAction?: () => void;
  extraContent?: React.ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title = 'No items found',
  description = 'Try selecting another category or refining your search parameters.',
  icon,
  actionText,
  onAction,
  extraContent,
}) => {
  return (
    <div className="glass-card animate-fade-in" style={{
      padding: '3rem 1.5rem',
      textAlign: 'center',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      width: '100%',
      margin: '1.5rem auto',
    }}>
      <div style={{
        width: '3.75rem',
        height: '3.75rem',
        borderRadius: 'var(--radius-lg)',
        backgroundColor: 'rgba(99, 102, 241, 0.1)',
        border: '1px solid rgba(99, 102, 241, 0.2)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: '1.25rem',
        color: '#818cf8',
      }}>
        {icon || <Inbox size={30} />}
      </div>

      <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem', color: '#ffffff' }}>
        {title}
      </h3>
      <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', maxWidth: '460px', marginBottom: (actionText || extraContent) ? '1.5rem' : 0 }}>
        {description}
      </p>

      {extraContent && (
        <div style={{ width: '100%', maxWidth: '480px', marginBottom: actionText ? '1.25rem' : 0 }}>
          {extraContent}
        </div>
      )}

      {actionText && onAction && (
        <button onClick={onAction} className="btn btn-primary">
          {actionText}
        </button>
      )}
    </div>
  );
};
