"""Standalone smoke tests for the inventory prediction service."""

import json

from inventory_service import (
    get_inventory_prediction,
    get_all_inventory_predictions,
)


REQUIRED_FIELDS = {
    "product_id",
    "product_name",
    "category",
    "current_stock",
    "predicted_7_day_demand",
    "required_stock_to_maintain",
    "stockout_probability",
    "recommended_order_quantity",
    "alert_needed",
    "alert_level",
    "model_metrics",
}


def main():
    for product_id in ("SKU001", "SKU010", "SKU040"):
        result = get_inventory_prediction(product_id)
        assert isinstance(result, dict)
        assert REQUIRED_FIELDS.issubset(result)
        assert result["predicted_7_day_demand"] >= 0
        assert result["required_stock_to_maintain"] >= result["predicted_7_day_demand"]
        assert 0.0 <= result["stockout_probability"] <= 1.0
        assert result["recommended_order_quantity"] >= 0
        json.dumps(result)
        print(f"{product_id}: Demand={result['predicted_7_day_demand']}, ReqStock={result['required_stock_to_maintain']}")

    # Test custom trend override
    custom_pred = get_inventory_prediction("SKU001", trend_score=0.8)
    assert custom_pred["trend_score"] == 0.8
    assert custom_pred["required_stock_to_maintain"] > 0
    print(f"Custom trend test passed: trend={custom_pred['trend_score']}, req_stock={custom_pred['required_stock_to_maintain']}")

    # Test all-product batch prediction
    batch = get_all_inventory_predictions(default_trend_score=0.8)
    assert batch["total_skus"] == 40
    assert len(batch["predictions"]) == 40
    assert batch["total_required_stock"] > 0
    print(f"Batch prediction check passed: {batch['total_skus']} SKUs evaluated.")

    try:
        get_inventory_prediction("SKU999")
    except ValueError as error:
        assert str(error) == "Unknown product_id: SKU999"
        print(f"Invalid SKU check passed: {error}")
    else:
        raise AssertionError("SKU999 should raise ValueError")

    print("All inventory service checks passed successfully.")


if __name__ == "__main__":
    main()

