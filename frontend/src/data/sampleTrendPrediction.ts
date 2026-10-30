import type { TrendPredictResult, TrendingItem } from '../types/api';

/**
 * Sample trend prediction payload matching the exact output from POST /predict
 * for "artificial intelligence" and "machine learning".
 */
export const SAMPLE_TREND_PREDICTION: TrendPredictResult = {
  keywords: [
    "artificial intelligence",
    "machine learning"
  ],
  topic: "artificial intelligence OR machine learning",
  rows_collected: 117,
  current_trend_score: 163.87,
  predicted_in_15_days: 4.08,
  change_15d_pct: -97.5,
  "change_15d_%": -97.5,
  indication_15d: "📉 Declining",
  predicted_in_30_days: 9.87,
  change_30d_pct: -94,
  "change_30d_%": -94,
  indication_30d: "📉 Declining",
  model_source: "database",
  aggregation_summary: {
    posts_count: 117,
    observed_days: 8,
    start_date: "2026-10-05",
    end_date: "2026-10-12",
    continuous_days: 8,
    line1: "Aggregated 117 tweets into 8 observed days (2026-10-05 to 2026-10-12)",
    line2: "Reindexed to 8 continuous days for modeling"
  },
  summary_table: [
    {
      topic: "artificial intelligence OR machine learning",
      current_trend_score: 163.87,
      predicted_in_15_days: 4.08,
      "change_15d_%": -97.5,
      indication_15d: "📉 Declining",
      predicted_in_30_days: 9.87,
      "change_30d_%": -94,
      indication_30d: "📉 Declining"
    }
  ],
  observed: [
    {
      date: "2026-10-05",
      observed_trend_score: 85.72,
      trend_score: 85.72
    },
    {
      date: "2026-10-06",
      observed_trend_score: 141.53,
      trend_score: 141.53
    },
    {
      date: "2026-10-07",
      observed_trend_score: 129.87,
      trend_score: 129.87
    },
    {
      date: "2026-10-08",
      observed_trend_score: 182.14,
      trend_score: 182.14
    },
    {
      date: "2026-10-09",
      observed_trend_score: 186.17,
      trend_score: 186.17
    },
    {
      date: "2026-10-10",
      observed_trend_score: 184.34,
      trend_score: 184.34
    },
    {
      date: "2026-10-11",
      observed_trend_score: 167.68,
      trend_score: 167.68
    },
    {
      date: "2026-10-12",
      observed_trend_score: 163.87,
      trend_score: 163.87
    }
  ],
  forecast: [
    {
      day: 1,
      date: "2026-10-13",
      predicted_score: 59.79,
      predicted_trend_score: 59.79
    },
    {
      day: 2,
      date: "2026-10-14",
      predicted_score: 51.46,
      predicted_trend_score: 51.46
    },
    {
      day: 3,
      date: "2026-10-15",
      predicted_score: 44.75,
      predicted_trend_score: 44.75
    },
    {
      day: 4,
      date: "2026-10-16",
      predicted_score: 36.64,
      predicted_trend_score: 36.64
    },
    {
      day: 5,
      date: "2026-10-17",
      predicted_score: 27.98,
      predicted_trend_score: 27.98
    },
    {
      day: 6,
      date: "2026-10-18",
      predicted_score: 18.99,
      predicted_trend_score: 18.99
    },
    {
      day: 7,
      date: "2026-10-19",
      predicted_score: 12.52,
      predicted_trend_score: 12.52
    },
    {
      day: 8,
      date: "2026-10-20",
      predicted_score: 7.5,
      predicted_trend_score: 7.5
    },
    {
      day: 9,
      date: "2026-10-21",
      predicted_score: 5.73,
      predicted_trend_score: 5.73
    },
    {
      day: 10,
      date: "2026-10-22",
      predicted_score: 4.81,
      predicted_trend_score: 4.81
    },
    {
      day: 11,
      date: "2026-10-23",
      predicted_score: 4.65,
      predicted_trend_score: 4.65
    },
    {
      day: 12,
      date: "2026-10-24",
      predicted_score: 4.42,
      predicted_trend_score: 4.42
    },
    {
      day: 13,
      date: "2026-10-25",
      predicted_score: 4.15,
      predicted_trend_score: 4.15
    },
    {
      day: 14,
      date: "2026-10-26",
      predicted_score: 4.09,
      predicted_trend_score: 4.09
    },
    {
      day: 15,
      date: "2026-10-27",
      predicted_score: 4.08,
      predicted_trend_score: 4.08
    },
    {
      day: 16,
      date: "2026-10-28",
      predicted_score: 4.04,
      predicted_trend_score: 4.04
    },
    {
      day: 17,
      date: "2026-10-29",
      predicted_score: 4.1,
      predicted_trend_score: 4.1
    },
    {
      day: 18,
      date: "2026-10-30",
      predicted_score: 4.21,
      predicted_trend_score: 4.21
    },
    {
      day: 19,
      date: "2026-10-31",
      predicted_score: 4.19,
      predicted_trend_score: 4.19
    },
    {
      day: 20,
      date: "2026-11-01",
      predicted_score: 4.18,
      predicted_trend_score: 4.18
    },
    {
      day: 21,
      date: "2026-11-02",
      predicted_score: 4.34,
      predicted_trend_score: 4.34
    },
    {
      day: 22,
      date: "2026-11-03",
      predicted_score: 4.65,
      predicted_trend_score: 4.65
    },
    {
      day: 23,
      date: "2026-11-04",
      predicted_score: 5.13,
      predicted_trend_score: 5.13
    },
    {
      day: 24,
      date: "2026-11-05",
      predicted_score: 5.78,
      predicted_trend_score: 5.78
    },
    {
      day: 25,
      date: "2026-11-06",
      predicted_score: 6.39,
      predicted_trend_score: 6.39
    },
    {
      day: 26,
      date: "2026-11-07",
      predicted_score: 7.06,
      predicted_trend_score: 7.06
    },
    {
      day: 27,
      date: "2026-11-08",
      predicted_score: 7.6,
      predicted_trend_score: 7.6
    },
    {
      day: 28,
      date: "2026-11-09",
      predicted_score: 8.28,
      predicted_trend_score: 8.28
    },
    {
      day: 29,
      date: "2026-11-10",
      predicted_score: 9.15,
      predicted_trend_score: 9.15
    },
    {
      day: 30,
      date: "2026-11-11",
      predicted_score: 9.87,
      predicted_trend_score: 9.87
    }
  ],
  plot_image: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABZYAAAKKCAYAAACu3Q5pAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjExLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlcelbwAAAAlwSFlzAAAT/gAAE/4BB5Q5hAABAABJREFUeJzs3Qd4U+UXx/HTyd5bNggislUERRAHKDhAcePEP24cuHDvjYp7I4hbce8FbkVFGSJ7yt57tM3/+b3lljRNS5O2uWn7/TzPJSW5Sd68d+Tm3HPPmxAIBAIGAAAAAAAAAEA+JeZ3RgAAAAAAAAAACCwDAAAAAAAAACJGxjIAAAAAAAAAICIElgEAAAAAAAAAESGwDAAAAAAAAACICIFlAAAAAAAAAEBECCwDAAAAAAAAACJCYBkAAAAAAAAAEBECywAAAAAAAACAiBBYBgAAAAAAAABEhMAyAAAAAAAAACAiBJYBAAAAAAAAABEhsAwAAAAAAAAAiAiBZQAAAAAAAABARAgsAwAAAAAAAAAiQmAZAAAAAAAAABARAssAAAAAAAAAgIgQWAYAAAAAAAAARITAMrDTli1b7IILLrBrr702bJ9kZGTYW2+9ZTfddJNdeOGFbt60tDSbOHGi+/uFF14o0vcv6tcqzPcvarm1NdL7AT+tWLHCrZe33nprRM8rDutzcWhjQXzzzTfu87333nt+NwWFaNGiRW653nPPPfRrCfDFF1+45Tlt2jS/m1Jsff75564PP/roI7+bgiKm3zRa1pdffjl9jUJbp9555x275pprsn47r1+/vkCv+c8//7jXefLJJ0vcscf9999vpU1hffYJEya41/n5558LrW2ITEIgEAhE+BygRFq7dq1Vq1bN6tSpY0uXLs322KZNm+zwww+3X375JUfwRAfbJ554op188sn2+uuvF8n7x+K1CvP9i1pubY30fkRvzZo19uWXX9rUqVPd31WrVrW9997bevXqZTVq1Mj1eRdffLGl6dn/T8hIcHKli1rdevWtQMPPNC6devm7isNZs2aZS1atLC99trL/v3333w/rzisz8WhjQXxyCOP2BVXXOEC5/fee6+VFm+88YZ9++23ET0nKSnJnnjiCSsO/vrrL+vYsaPtu+++9vvvv+frOTrpfNFFF+12vuuvv94aNWpURK1Efmzbts1atWplycnJbv+q9RCR0/5t2LBhdsstt0R8EhTFy9atW61cuXJWoUIF27hxo9/NQTG3Y8cOO+KII2z8+PHZ7l+yZIk75o/WZ599ZkcddZT17du3xJzw8o49DjjggByxhpKusD67YjVNmza1+vXr259//llqfkvGk2S/GwDEgs6O6mypAh3RZCI9//zzbmdXr149dzasdu3a7v6UlJQiaG3JUNA+R/weKOrHpQJrmzdvzvF4mTJlbMiQIXbHHXe4v0M988wz2QLLoRTQGTt2LAEYIA59//33bhsuqYHlaCiwnJ8+Oe+889iv5cNvv/1mL774onXt2tXOOuusqJfLY489ZvPmzbOXXnqJoHIJpsCSpuOOO84Fm0r6+wKRXl315ptvugDvCSecENPOU7KVgspKNtHJ1z322MPdX6VKlZi2A9nNnTvX7rvvPttzzz3tqquuKjHdoxNi+jxK+BgzZoydccYZfjep1CGwjFJBATD98NNZrNyCnOXLl7ennnrK3Yb7MS133323nX322dke69Spk3uedtAFkdf7x0Jhv39++jxW/O7bkpQBpgyBr7/+Omvd1w8qnWhZuXKlu+z4119/tQceeMDdKqtA2S/h3HDDDdagQQP3tzKelR2osg"
};

export function createSampleTrendPrediction(
  topic: string,
  keywords: string[],
  currentScore: number,
  change15Pct: number,
  change30Pct: number,
  postsCount: number = 85
): TrendPredictResult {
  const d15 = Number((currentScore * (1 + change15Pct / 100)).toFixed(2));
  const d30 = Number((currentScore * (1 + change30Pct / 100)).toFixed(2));

  const indication15 = change15Pct >= 15 ? '📈 Rising' : change15Pct <= -15 ? '📉 Declining' : '➡️ Stable';
  const indication30 = change30Pct >= 15 ? '📈 Rising' : change30Pct <= -15 ? '📉 Declining' : '➡️ Stable';

  const baseDate = new Date();

  // 8 Observed Days
  const observed = Array.from({ length: 8 }, (_, i) => {
    const d = new Date(baseDate);
    d.setDate(baseDate.getDate() - (7 - i));
    const dateStr = d.toISOString().slice(0, 10);
    const progress = i / 7;
    const histScore = Number((currentScore * (0.85 + 0.15 * progress) + Math.sin(i * 1.5) * 3).toFixed(2));
    const finalScore = i === 7 ? currentScore : histScore;
    return {
      date: dateStr,
      observed_trend_score: finalScore,
      trend_score: finalScore,
    };
  });

  // 30 Forecast Days
  const forecast = Array.from({ length: 30 }, (_, i) => {
    const dayNum = i + 1;
    const d = new Date(baseDate);
    d.setDate(baseDate.getDate() + dayNum);
    const dateStr = d.toISOString().slice(0, 10);

    let score: number;
    if (dayNum <= 15) {
      const t = dayNum / 15;
      score = currentScore + (d15 - currentScore) * Math.sin((t * Math.PI) / 2);
    } else {
      const t = (dayNum - 15) / 15;
      score = d15 + (d30 - d15) * Math.sin((t * Math.PI) / 2);
    }
    const wobble = Math.sin(dayNum * 0.9) * 0.6;
    const rounded = Number((score + wobble).toFixed(2));

    return {
      day: dayNum,
      date: dateStr,
      predicted_score: rounded,
      predicted_trend_score: rounded,
    };
  });

  return {
    keywords,
    topic,
    rows_collected: postsCount,
    current_trend_score: currentScore,
    predicted_in_15_days: d15,
    change_15d_pct: change15Pct,
    'change_15d_%': change15Pct,
    indication_15d: indication15,
    predicted_in_30_days: d30,
    change_30d_pct: change30Pct,
    'change_30d_%': change30Pct,
    indication_30d: indication30,
    model_source: 'database',
    aggregation_summary: {
      posts_count: postsCount,
      observed_days: 8,
      start_date: observed[0].date,
      end_date: observed[7].date,
      continuous_days: 8,
      line1: `Aggregated ${postsCount} social signals into 8 observed days (${observed[0].date} to ${observed[7].date})`,
      line2: 'Reindexed to 8 continuous days for gradient boosted forecasting',
    },
    summary_table: [
      {
        topic,
        current_trend_score: currentScore,
        predicted_in_15_days: d15,
        'change_15d_%': change15Pct,
        indication_15d: indication15,
        predicted_in_30_days: d30,
        'change_30d_%': change30Pct,
        indication_30d: indication30,
      },
    ],
    observed,
    forecast,
  };
}

export const SME_SAMPLE_PREDICTIONS: Record<string, TrendPredictResult> = {
  cafes: createSampleTrendPrediction(
    'matcha latte',
    ['matcha latte', 'iced matcha', 'oat milk latte'],
    58.4,
    26.8,
    48.5,
    88
  ),
  restaurants: createSampleTrendPrediction(
    'biryani & cloud kitchens',
    ['biryani', 'dum biryani', 'kebab'],
    148.2,
    4.5,
    8.2,
    142
  ),
  boutiques: createSampleTrendPrediction(
    'co-ord sets & festive wear',
    ['co-ord set', 'ethnic wear', 'linen co-ord'],
    74.5,
    28.4,
    56.2,
    96
  ),
  salons: createSampleTrendPrediction(
    'keratin treatment & hair spa',
    ['keratin treatment', 'hair botox', 'scalp detox'],
    62.1,
    7.2,
    14.6,
    74
  ),
  gyms: createSampleTrendPrediction(
    'pilates & functional fitness',
    ['pilates', 'reformer pilates', 'hiit studio'],
    84.7,
    24.2,
    51.8,
    110
  ),
  gifts: createSampleTrendPrediction(
    'scented candles & artisanal decor',
    ['scented candles', 'soy candle', 'home fragrance'],
    49.3,
    31.5,
    64.2,
    68
  ),
  electronics: createSampleTrendPrediction(
    'wireless earbuds & anc accessories',
    ['wireless earbuds', 'anc headphones', 'tws earbuds'],
    114.8,
    3.8,
    7.5,
    125
  ),
  d2c: createSampleTrendPrediction(
    'handcrafted pottery & ceramics',
    ['handcrafted pottery', 'ceramic mugs', 'terracotta'],
    38.6,
    21.4,
    46.8,
    55
  ),
};

// Aliases for backend keys
SME_SAMPLE_PREDICTIONS['cafe'] = SME_SAMPLE_PREDICTIONS.cafes;
SME_SAMPLE_PREDICTIONS['restaurant'] = SME_SAMPLE_PREDICTIONS.restaurants;
SME_SAMPLE_PREDICTIONS['boutique'] = SME_SAMPLE_PREDICTIONS.boutiques;
SME_SAMPLE_PREDICTIONS['salon'] = SME_SAMPLE_PREDICTIONS.salons;
SME_SAMPLE_PREDICTIONS['gym'] = SME_SAMPLE_PREDICTIONS.gyms;
SME_SAMPLE_PREDICTIONS['gift_shop'] = SME_SAMPLE_PREDICTIONS.gifts;
SME_SAMPLE_PREDICTIONS['electronics'] = SME_SAMPLE_PREDICTIONS.electronics;
SME_SAMPLE_PREDICTIONS['d2c'] = SME_SAMPLE_PREDICTIONS.d2c;

export const SME_FALLBACK_TREND_ITEMS: Record<string, TrendingItem[]> = {
  cafes: [
    {
      item_name: 'Dubai Kunafa Chocolate Pistachio Bar',
      category: 'Dessert',
      trend_status: 'Explosive Growth',
      estimated_growth_pct: 145,
      urgency_score: 95,
      commercial_appeal: 'Middle Eastern pastry fusion with chocolate and roasted pistachio praline.',
      local_relevance: 'High (Bangalore Local Favorite)',
    },
    {
      item_name: 'Iced Matcha Latte',
      category: 'Beverage',
      trend_status: 'Rising',
      estimated_growth_pct: 88,
      urgency_score: 90,
      commercial_appeal: 'Ceremonial green tea with oat milk and cold foam layer.',
      local_relevance: 'High (Bangalore Local Favorite)',
    },
    {
      item_name: 'Cold Brew Coffee Concentrate',
      category: 'Beverage',
      trend_status: 'Rising',
      estimated_growth_pct: 62,
      urgency_score: 78,
      commercial_appeal: '18-hour slow steeped specialty coffee for home and cafe enjoyment.',
      local_relevance: 'High',
    },
    {
      item_name: 'Matcha Green Tea Powder',
      category: 'Beverage Ingredient',
      trend_status: 'Rising',
      estimated_growth_pct: 54,
      urgency_score: 72,
      commercial_appeal: 'Ceremonial Japanese Uji green tea powder for wellness drinks.',
      local_relevance: 'Medium',
    },
    {
      item_name: 'Bottomless Brunch',
      category: 'Dining Concept',
      trend_status: 'Steady / Evergreen',
      estimated_growth_pct: 41,
      urgency_score: 68,
      commercial_appeal: 'Weekend communal dining with artisanal toasts and craft brews.',
      local_relevance: 'High (Bangalore Local Favorite)',
    },
    {
      item_name: 'Taro Boba Milk Tea',
      category: 'Beverage',
      trend_status: 'Steady / Evergreen',
      estimated_growth_pct: 38,
      urgency_score: 60,
      commercial_appeal: 'Chewy brown sugar tapioca pearls in rich purple taro milk.',
      local_relevance: 'Medium',
    },
  ],
  restaurants: [
    {
      item_name: 'Donne Biryani',
      category: 'Main Course',
      trend_status: 'Explosive Growth',
      estimated_growth_pct: 135,
      urgency_score: 94,
      commercial_appeal: 'Heritage seeraga samba rice cooked with fresh mint, coriander, and spiced meat.',
      local_relevance: 'High (Bangalore Local Favorite)',
    },
    {
      item_name: 'Neapolitan Sourdough Pizza',
      category: 'Main Course',
      trend_status: 'Rising',
      estimated_growth_pct: 112,
      urgency_score: 89,
      commercial_appeal: 'Wood-fired sourdough crust topped with burrata and san marzano tomatoes.',
      local_relevance: 'High',
    },
    {
      item_name: 'Smash Cheeseburger',
      category: 'Fast Casual',
      trend_status: 'Rising',
      estimated_growth_pct: 85,
      urgency_score: 84,
      commercial_appeal: 'Crispy seared edges, caramelized onions, and melted sharp cheddar in brioche.',
      local_relevance: 'High (Bangalore Local Favorite)',
    },
    {
      item_name: 'Tonkotsu Ramen Bowl',
      category: 'Main Course',
      trend_status: 'Rising',
      estimated_growth_pct: 76,
      urgency_score: 80,
      commercial_appeal: 'Rich 12-hour collagen broth with springy noodles and torched chashu.',
      local_relevance: 'High',
    },
    {
      item_name: 'Ghee Podi Thatte Idli',
      category: 'Breakfast / Snack',
      trend_status: 'Rising',
      estimated_growth_pct: 68,
      urgency_score: 75,
      commercial_appeal: 'Ultra-soft steamed rice cakes drenched in pure desi ghee and spicy podi.',
      local_relevance: 'High (Bangalore Local Favorite)',
    },
    {
      item_name: 'Crispy Korean Fried Chicken',
      category: 'Snack / Main Course',
      trend_status: 'Steady / Evergreen',
      estimated_growth_pct: 58,
      urgency_score: 72,
      commercial_appeal: 'Double-fried crunchy chicken wings with spicy gochujang glaze.',
      local_relevance: 'Medium',
    },
  ],
  boutiques: [
    {
      item_name: 'Linen Co-ord Sets',
      category: 'Apparel',
      trend_status: 'Explosive Growth',
      estimated_growth_pct: 155,
      urgency_score: 96,
      commercial_appeal: 'Breathable, two-piece minimalist linen styling for effortless urban chic.',
      local_relevance: 'High (Bangalore Local Favorite)',
    },
    {
      item_name: 'Chikankari Modal Kurtis',
      category: 'Ethnic Wear',
      trend_status: 'Explosive Growth',
      estimated_growth_pct: 128,
      urgency_score: 92,
      commercial_appeal: 'Soft flowy modal fabric with intricate Lucknowi hand-embroidery in pastels.',
      local_relevance: 'High',
    },
    {
      item_name: 'Oversized Graphic Streetwear',
      category: 'Casual Apparel',
      trend_status: 'Rising',
      estimated_growth_pct: 110,
      urgency_score: 88,
      commercial_appeal: '240 GSM heavyweight drop-shoulder tees with typography backprints.',
      local_relevance: 'High (Bangalore Local Favorite)',
    },
    {
      item_name: 'Pastel Organza Sarees',
      category: 'Festive Wear',
      trend_status: 'Rising',
      estimated_growth_pct: 85,
      urgency_score: 82,
      commercial_appeal: 'Sheer lightweight organza sarees with scalloped borders for festive celebrations.',
      local_relevance: 'High',
    },
    {
      item_name: 'Washed Cargo Parachute Pants',
      category: 'Bottomwear',
      trend_status: 'Rising',
      estimated_growth_pct: 74,
      urgency_score: 78,
      commercial_appeal: 'Utilitarian baggy parachute pants with drawstrings and cargo utility pockets.',
      local_relevance: 'Medium',
    },
    {
      item_name: 'Vintage Thrifted Denim Jackets',
      category: 'Sustainable Fashion',
      trend_status: 'Rising',
      estimated_growth_pct: 68,
      urgency_score: 75,
      commercial_appeal: 'One-of-a-kind upcycled acid-wash jackets popular in sustainable fashion circles.',
      local_relevance: 'High (Bangalore Local Favorite)',
    },
  ],
  salons: [
    {
      item_name: 'Korean Glass Skin Facial',
      category: 'Skincare Treatment',
      trend_status: 'Explosive Growth',
      estimated_growth_pct: 165,
      urgency_score: 98,
      commercial_appeal: 'Multi-step peeling, peptide infusion, and LED therapy for a luminous dewy finish.',
      local_relevance: 'High (Bangalore Local Favorite)',
    },
    {
      item_name: 'Keratin Smoothing Treatment',
      category: 'Haircare Service',
      trend_status: 'Explosive Growth',
      estimated_growth_pct: 120,
      urgency_score: 90,
      commercial_appeal: 'Formaldehyde-free protein smoothing that eliminates frizz for up to 4 months.',
      local_relevance: 'High',
    },
    {
      item_name: 'Sun-Kissed Balayage Highlights',
      category: 'Hair Color',
      trend_status: 'Rising',
      estimated_growth_pct: 105,
      urgency_score: 88,
      commercial_appeal: 'Seamless dimensional highlights blending into caramel and honey tones.',
      local_relevance: 'High',
    },
    {
      item_name: 'Chrome Glazed Donut Nails',
      category: 'Nail Art',
      trend_status: 'Rising',
      estimated_growth_pct: 95,
      urgency_score: 85,
      commercial_appeal: 'Iridescent pearl chrome powder on milky gel extensions.',
      local_relevance: 'High (Bangalore Local Favorite)',
    },
    {
      item_name: 'Scalp Detox Head Spa',
      category: 'Wellness / Haircare',
      trend_status: 'Rising',
      estimated_growth_pct: 82,
      urgency_score: 82,
      commercial_appeal: 'Japanese head-spa steam circulation and acupressure therapy for scalp wellness.',
      local_relevance: 'High',
    },
    {
      item_name: 'Brow Lamination & Tint',
      category: 'Grooming',
      trend_status: 'Rising',
      estimated_growth_pct: 72,
      urgency_score: 76,
      commercial_appeal: 'Feathered upward brow styling for fuller, effortless natural brows.',
      local_relevance: 'Medium',
    },
  ],
  gyms: [
    {
      item_name: 'Reformer Pilates Sculpt',
      category: 'Fitness Class',
      trend_status: 'Explosive Growth',
      estimated_growth_pct: 170,
      urgency_score: 98,
      commercial_appeal: 'Low-impact spring-resistance core carriage training and lean body toning.',
      local_relevance: 'High (Bangalore Local Favorite)',
    },
    {
      item_name: 'Functional HIIT & Hyrox Conditioning',
      category: 'Athletic Training',
      trend_status: 'Explosive Growth',
      estimated_growth_pct: 115,
      urgency_score: 89,
      commercial_appeal: 'High-intensity sled push, rowing, and functional athletic endurance circuits.',
      local_relevance: 'High',
    },
    {
      item_name: 'Sound Bath & Yin Yoga',
      category: 'Mindfulness / Yoga',
      trend_status: 'Rising',
      estimated_growth_pct: 98,
      urgency_score: 85,
      commercial_appeal: 'Restorative deep fascia stretch paired with acoustic Tibetan singing bowl therapy.',
      local_relevance: 'High',
    },
    {
      item_name: 'Pickleball League Training',
      category: 'Racquet Sports',
      trend_status: 'Explosive Growth',
      estimated_growth_pct: 140,
      urgency_score: 92,
      commercial_appeal: 'Fast-paced, accessible social paddle sport booming across weekend clubs.',
      local_relevance: 'High (Bangalore Local Favorite)',
    },
    {
      item_name: 'Calisthenics & Joint Mobility',
      category: 'Bodyweight Fitness',
      trend_status: 'Rising',
      estimated_growth_pct: 78,
      urgency_score: 80,
      commercial_appeal: 'Bodyweight strength control on gymnastics rings and active joint recovery.',
      local_relevance: 'Medium',
    },
    {
      item_name: 'Recovery Foam Rolling & Cold Plunge',
      category: 'Active Recovery',
      trend_status: 'Rising',
      estimated_growth_pct: 88,
      urgency_score: 84,
      commercial_appeal: 'Contrast cold immersion and percussion therapy for accelerated muscle recovery.',
      local_relevance: 'High',
    },
  ],
  gifts: [
    {
      item_name: 'Artisanal Soy Scented Candles',
      category: 'Home Fragrance',
      trend_status: 'Explosive Growth',
      estimated_growth_pct: 148,
      urgency_score: 94,
      commercial_appeal: 'Non-toxic hand-poured soy wax candles with crackling wood wicks in amber jars.',
      local_relevance: 'High',
    },
    {
      item_name: 'Curated Celebration Gift Hampers',
      category: 'Gifting Concept',
      trend_status: 'Explosive Growth',
      estimated_growth_pct: 130,
      urgency_score: 91,
      commercial_appeal: 'Bespoke wooden hamper boxes with gourmet sweets, teas, and customized keepsakes.',
      local_relevance: 'High',
    },
    {
      item_name: 'Custom Resin Art Nameplates',
      category: 'Personalized Decor',
      trend_status: 'Rising',
      estimated_growth_pct: 95,
      urgency_score: 86,
      commercial_appeal: 'Epoxy resin nameplates with gold leafing and botanical accents for modern homes.',
      local_relevance: 'High (Bangalore Local Favorite)',
    },
    {
      item_name: 'Geometric Succulent Terrariums',
      category: 'Botanical Decor',
      trend_status: 'Rising',
      estimated_growth_pct: 82,
      urgency_score: 80,
      commercial_appeal: 'Faceted glass vessels holding living succulents and colored sand layers.',
      local_relevance: 'High',
    },
    {
      item_name: 'Minimalist Ribbed Ceramic Vases',
      category: 'Home Decor',
      trend_status: 'Rising',
      estimated_growth_pct: 74,
      urgency_score: 76,
      commercial_appeal: 'Nordic-style matte ceramic vases designed for pampas grass and table decor.',
      local_relevance: 'Medium',
    },
    {
      item_name: 'Wax Seal Journaling Stationery Kits',
      category: 'Lifestyle / Craft',
      trend_status: 'Rising',
      estimated_growth_pct: 62,
      urgency_score: 70,
      commercial_appeal: 'Brass stamps, melting spoons, and wax beads for aesthetic gift wrap and letters.',
      local_relevance: 'Medium',
    },
  ],
  electronics: [
    {
      item_name: 'Active Noise Cancelling Wireless Earbuds',
      category: 'Audio Tech',
      trend_status: 'Explosive Growth',
      estimated_growth_pct: 135,
      urgency_score: 92,
      commercial_appeal: 'Hybrid noise cancellation with transparency audio and multipoint Bluetooth pairing.',
      local_relevance: 'High',
    },
    {
      item_name: 'Magnetic Wireless Power Bank',
      category: 'Mobile Accessories',
      trend_status: 'Rising',
      estimated_growth_pct: 118,
      urgency_score: 88,
      commercial_appeal: 'Snap-on slim MagSafe compatible power bank with built-in phone kickstand.',
      local_relevance: 'High',
    },
    {
      item_name: 'AMOLED Smart Fitness Band',
      category: 'Wearable Tech',
      trend_status: 'Rising',
      estimated_growth_pct: 95,
      urgency_score: 84,
      commercial_appeal: 'Continuous SpO2, sleep tracking, and heart-rate monitoring with 14-day battery.',
      local_relevance: 'High',
    },
    {
      item_name: 'RGB Hot-Swappable Mechanical Keyboard',
      category: 'Computer Peripherals',
      trend_status: 'Rising',
      estimated_growth_pct: 88,
      urgency_score: 82,
      commercial_appeal: 'Compact layout with lubricated switches, sound dampening foam, and wireless tri-mode.',
      local_relevance: 'High (Bangalore Local Favorite)',
    },
    {
      item_name: '65W GaN Compact Fast Charger',
      category: 'Charging Tech',
      trend_status: 'Rising',
      estimated_growth_pct: 78,
      urgency_score: 79,
      commercial_appeal: 'Pocket-sized charger simultaneously powering laptop, tablet, and smartphone.',
      local_relevance: 'High',
    },
    {
      item_name: 'Smart Air Quality & Desk Ambient Monitor',
      category: 'Smart Home Tech',
      trend_status: 'Rising',
      estimated_growth_pct: 64,
      urgency_score: 72,
      commercial_appeal: 'Real-time indoor PM2.5 and CO2 monitor with circadian ambient desk lighting.',
      local_relevance: 'Medium',
    },
  ],
  d2c: [
    {
      item_name: 'Small-Batch Handcrafted Pottery Mugs',
      category: 'Artisan Kitchenware',
      trend_status: 'Explosive Growth',
      estimated_growth_pct: 150,
      urgency_score: 95,
      commercial_appeal: 'Wheel-thrown ceramic stoneware mugs with speckled reactive glazes.',
      local_relevance: 'High (Bangalore Local Favorite)',
    },
    {
      item_name: 'Organic Whipped Shea Body Butter',
      category: 'Clean Beauty',
      trend_status: 'Explosive Growth',
      estimated_growth_pct: 125,
      urgency_score: 90,
      commercial_appeal: 'Waterless whipped raw shea and mango butter for deep moisture.',
      local_relevance: 'High',
    },
    {
      item_name: 'Dainty Freshwater Pearl Beaded Jewelry',
      category: 'Handmade Accessories',
      trend_status: 'Rising',
      estimated_growth_pct: 108,
      urgency_score: 87,
      commercial_appeal: 'Irregular baroque pearls and anti-tarnish 18k gold beads on handmade chokers.',
      local_relevance: 'High',
    },
    {
      item_name: 'Eco-Friendly Heavy Canvas Tote Bags',
      category: 'Sustainable Accessories',
      trend_status: 'Rising',
      estimated_growth_pct: 92,
      urgency_score: 82,
      commercial_appeal: 'Durable 16oz cotton canvas totes with laptop compartment and bottle sleeves.',
      local_relevance: 'High (Bangalore Local Favorite)',
    },
    {
      item_name: 'Botanical Scented Soy Candles',
      category: 'Home Decor',
      trend_status: 'Rising',
      estimated_growth_pct: 85,
      urgency_score: 80,
      commercial_appeal: 'Hand-poured coconut-soy wax decorated with dried botanicals and raw crystals.',
      local_relevance: 'Medium',
    },
    {
      item_name: 'Cold-Pressed Herbal Face Elixir',
      category: 'Organic Skincare',
      trend_status: 'Rising',
      estimated_growth_pct: 76,
      urgency_score: 78,
      commercial_appeal: 'Facial oil infused with Kashmiri saffron, rosehip, and kumkumadi botanicals.',
      local_relevance: 'High',
    },
  ],
};

// Aliases for backend keys
SME_FALLBACK_TREND_ITEMS['cafe'] = SME_FALLBACK_TREND_ITEMS.cafes;
SME_FALLBACK_TREND_ITEMS['restaurant'] = SME_FALLBACK_TREND_ITEMS.restaurants;
SME_FALLBACK_TREND_ITEMS['boutique'] = SME_FALLBACK_TREND_ITEMS.boutiques;
SME_FALLBACK_TREND_ITEMS['salon'] = SME_FALLBACK_TREND_ITEMS.salons;
SME_FALLBACK_TREND_ITEMS['gym'] = SME_FALLBACK_TREND_ITEMS.gyms;
SME_FALLBACK_TREND_ITEMS['gift_shop'] = SME_FALLBACK_TREND_ITEMS.gifts;
SME_FALLBACK_TREND_ITEMS['electronics'] = SME_FALLBACK_TREND_ITEMS.electronics;
SME_FALLBACK_TREND_ITEMS['d2c'] = SME_FALLBACK_TREND_ITEMS.d2c;

