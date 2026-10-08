# Inventory + Demand Prediction Module

## Purpose

This module provides:

- 7-day demand prediction
- 3-day stockout risk prediction
- Reorder recommendation
- Inventory alerts

## Dataset

The current expanded dataset contains 40 SKUs, 90 days, and 3,600 sales records.

## Models

Demand: `models/demand_regressor_expanded.pkl`  
Stockout: `models/stockout_model_expanded.pkl`

These models were trained and evaluated on the current expanded dataset. They
are baseline/initial production models; their predictions are not guaranteed to
be perfect.

Evaluation on the current test split:

- Demand MAE: 14.97
- Demand RMSE: 23.40
- Demand R²: 0.429
- Naive demand baseline MAE: 20.79
- Stockout accuracy: 0.810
- Stockout precision: 0.610
- Stockout recall: 0.888
- Stockout F1: 0.723

## Public Interface

```python
from src.inventory_service import get_inventory_prediction

result = get_inventory_prediction("SKU001")
print(result)
```

The values below illustrate the response shape only; prediction values depend
on the current dataset and models.

```python
{
    "product_id": "SKU001",
    "product_name": "Example product",
    "category": "Example category",
    "current_stock": 60,
    "predicted_7_day_demand": 25,
    "stockout_probability": 0.06,
    "recommended_order_quantity": 0,
    "alert_needed": False,
    "alert_level": "NONE"
}
```

## Input

`product_id`, for example `SKU001`.

## Output

The JSON-serializable result contains:

- `product_id`
- `product_name`
- `category`
- `current_stock`
- `predicted_7_day_demand`
- `stockout_probability`
- `recommended_order_quantity`
- `alert_needed`
- `alert_level`

## Integration Flow

```text
Team Application
      |
      | product_id
      v
inventory_service.py
      |
      +----> Expanded Demand Model
      |
      +----> Expanded Stockout Model
      |
      v
JSON-serializable inventory result
```

## Current Limitations

- No HTTP API yet
- No frontend included
- No advanced OCR-to-SKU matching
- SKU matching is expected to happen before calling the service
- Model retraining is a separate process
- Current models are baseline/initial production models, not guaranteed perfect
  predictions
