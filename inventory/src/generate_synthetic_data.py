import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

CATALOG_PATH = "inventory/data/product_catalog.csv"
OUTPUT_PATH = "inventory/data/sales_log.csv"
N_DAYS = 90

def generate_sales_log():
    catalog = pd.read_csv(CATALOG_PATH)
    start_date = datetime.today() - timedelta(days=N_DAYS)
    records = []

    for _, row in catalog.iterrows():
        product_id = row["product_id"]
        base_demand = np.random.uniform(2, 15)
        trend_boost_day = np.random.randint(30, 70)
        stock = int(row["current_stock"]) + 50

        for day in range(N_DAYS):
            date = start_date + timedelta(days=day)

            weekday_factor = 1.3 if date.weekday() >= 5 else 1.0
            trend_factor = (
                1.8
                if trend_boost_day <= day <= trend_boost_day + 10
                else 1.0
            )

            potential_demand = int(np.random.poisson(
                base_demand * weekday_factor * trend_factor
            ))

            daily_sales = min(potential_demand, stock)
            stock -= daily_sales

            if stock < 10 and np.random.random() < 0.3:
                stock += np.random.randint(40, 80)

            records.append({
                "product_id": product_id,
                "date": date.strftime("%Y-%m-%d"),
                "potential_demand": potential_demand,
                "units_sold": daily_sales,
                "stock_after": stock,
                "trend_score": round(trend_factor - 1.0, 2)
            })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False)

    print(
        f"Generated {len(df)} rows for "
        f"{catalog.shape[0]} products -> {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    generate_sales_log()
