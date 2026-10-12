import React, { useState, useEffect } from 'react';
import { Navbar, type PageId } from './components/common/Navbar';
import { Footer } from './components/common/Footer';
import { HomePage } from './pages/HomePage';
import { TrendPredictorPage } from './pages/TrendPredictorPage';
import { InfluencerPage } from './pages/InfluencerPage';
import { CreativeStudioPage } from './pages/CreativeStudioPage';
import { InventoryPage } from './pages/InventoryPage';
import { fetchHealth } from './services/api';
import type { HealthStatus } from './types/api';

export const App: React.FC = () => {
  // Sync page state with window.location.hash
  const getInitialPage = (): PageId => {
    const hash = window.location.hash.replace('#', '') as PageId;
    if (['home', 'trends', 'influencers', 'creative', 'inventory'].includes(hash)) {
      return hash;
    }
    return 'home';
  };

  const [currentPage, setCurrentPage] = useState<PageId>(getInitialPage());
  const [creativePresetPrompt, setCreativePresetPrompt] = useState<string>('');

  // Global Health state
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);

  // Poll health on mount and every 45s
  useEffect(() => {
    let isMounted = true;
    const checkHealth = async () => {
      try {
        const res = await fetchHealth();
        if (isMounted) setHealth(res);
      } catch (err) {
        if (isMounted) setHealth(null);
      } finally {
        if (isMounted) setHealthLoading(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 45000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // Update hash when page changes
  const handleNavigate = (page: PageId) => {
    setCurrentPage(page);
    window.location.hash = page;
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Allow passing a generated prompt from Trend Predictor directly to Creative Studio
  const handleNavigateToCreative = (promptPreset: string) => {
    setCreativePresetPrompt(promptPreset);
    handleNavigate('creative');
  };

  // Sync back-forward navigation with hashchange
  useEffect(() => {
    const handleHashChange = () => {
      const page = getInitialPage();
      setCurrentPage(page);
    };
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Navbar */}
      <Navbar
        currentPage={currentPage}
        onNavigate={handleNavigate}
        health={health}
        healthLoading={healthLoading}
      />

      {/* Main Page Container */}
      <main style={{ flex: 1 }}>
        {currentPage === 'home' && (
          <HomePage
            onNavigate={handleNavigate}
            health={health}
            healthLoading={healthLoading}
          />
        )}

        {currentPage === 'trends' && (
          <TrendPredictorPage
            onNavigateToCreative={handleNavigateToCreative}
          />
        )}

        {currentPage === 'influencers' && (
          <InfluencerPage />
        )}

        {currentPage === 'creative' && (
          <CreativeStudioPage
            initialPrompt={creativePresetPrompt}
          />
        )}

        {currentPage === 'inventory' && (
          <InventoryPage />
        )}
      </main>

      {/* Footer */}
      <Footer />
    </div>
  );
};

export default App;
