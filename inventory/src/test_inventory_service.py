"""Standalone smoke tests for the inventory prediction service."""

import json

from inventory_service import get_inventory_prediction


REQUIRED_FIELDS = {
    "product_id",
    "product_name",
    "category",
    "current_stock",
    "predicted_7_day_demand",
    "stockout_probability",
    "recommended_order_quantity",
    "alert_needed",
    "alert_level",
}


def main():
    for product_id in ("SKU001", "SKU010", "SKU040"):
        result = get_inventory_prediction(product_id)
        assert isinstance(result, dict)
        assert REQUIRED_FIELDS.issubset(result)
        assert result["predicted_7_day_demand"] >= 0
        assert 0.0 <= result["stockout_probability"] <= 1.0
        assert result["recommended_order_quantity"] >= 0
        json.dumps(result)
        print(f"{product_id}: {result}")

    try:
        get_inventory_prediction("SKU999")
    except ValueError as error:
        assert str(error) == "Unknown product_id: SKU999"
        print(f"Invalid SKU check passed: {error}")
    else:
        raise AssertionError("SKU999 should raise ValueError")

    print("All inventory service checks passed.")


if __name__ == "__main__":
    main()
