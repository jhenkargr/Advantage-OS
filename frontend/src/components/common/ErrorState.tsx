import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

interface ErrorStateProps {
  title?: string;
  message?: string;
  onRetry?: () => void;
  actionText?: string;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'Something went wrong',
  message = 'An unexpected error occurred while communicating with the service.',
  onRetry,
  actionText = 'Try Again',
}) => {
  return (
    <div className="glass-card animate-fade-in" style={{
      padding: '2.5rem 1.5rem',
      textAlign: 'center',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      borderColor: 'rgba(244, 63, 94, 0.3)',
      backgroundColor: 'rgba(244, 63, 94, 0.04)',
      maxWidth: '600px',
      margin: '1.5rem auto',
    }}>
      <div style={{
        width: '3.5rem',
        height: '3.5rem',
        borderRadius: '50%',
        backgroundColor: 'rgba(244, 63, 94, 0.12)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: '1rem',
        color: '#f43f5e',
      }}>
        <AlertCircle size={28} />
      </div>

      <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem', color: '#ffffff' }}>
        {title}
      </h3>
      <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', maxWidth: '420px', marginBottom: onRetry ? '1.5rem' : 0 }}>
        {message}
      </p>

      {onRetry && (
        <button
          onClick={onRetry}
          className="btn btn-secondary"
          style={{
            borderColor: 'rgba(244, 63, 94, 0.4)',
            color: '#ffffff',
          }}
        >
          <RefreshCw size={16} />
          {actionText}
        </button>
      )}
    </div>
  );
};
