#!/usr/bin/env python3
"""
Demand forecast + stock plan.

Inputs
  --sales     CSV with columns: date, item_sold, product_id, price
  --products  CSV with columns: product_id, product_name, current_price, category, stock_available

Output (one row per product)
  Product | Avg daily demand | 30-day forecast | Current stock | Safety stock |
  Reorder point | Target stock | Order qty now | Status

Method
  1. Sales rows are combined into a daily series per product (zeros on days with no sales).
  2. Demand is forecast with a 50/50 blend of
       - a global gradient-boosting model (Poisson loss, direct 1..H day horizons, 5 seeds), and
       - TSB, a method designed for intermittent (many zero days) demand.
  3. A growth correction is applied (weekly growth is a parameter because no trend was supplied).
  4. Stock rules:
       safety stock  = Z * daily_std * sqrt(lead_time)
       reorder point = daily_demand * lead_time + safety stock
       target stock  = daily_demand * (lead_time + horizon) + safety stock
       order qty     = target - stock, only when stock <= reorder point
  Products whose stock is 0 and whose sales stopped are treated as stocked out: the days after the
  last sale are excluded so the stock-out is not mistaken for low demand.

Usage
  python demand_model.py --sales sales_history.csv --products products.csv --out stock_plan
  python demand_model.py ... --lead-time 7 --service-level 0.95 --horizon 30 --growth 0.03 --lag 14 --uplift 0
  python demand_model.py ... --backtest        # also print the rolling back-test of the models

Needs: pandas numpy scikit-learn scipy openpyxl
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor

FEATS = ['k', 'dow', 'r7', 'r14', 'r28', 'rall', 'ratio', 'ds', 'nzm', 'nzrate28',
         'logp', 'pratio', 'cat', 'age']


# ----------------------------------------------------------------------------- data
def load(sales_path, products_path):
    if isinstance(products_path, pd.DataFrame):
        p = products_path.copy()
    else:
        p = pd.read_csv(products_path)

    if isinstance(sales_path, pd.DataFrame):
        s = sales_path.copy()
        s['date'] = pd.to_datetime(s['date'])
    else:
        s = pd.read_csv(sales_path, parse_dates=['date'])

    # Standardize column names if coming from models/other sources
    if 'current_price' not in p.columns and 'unit_price' in p.columns:
        p['current_price'] = p['unit_price']
    if 'stock_available' not in p.columns and 'current_stock' in p.columns:
        p['stock_available'] = p['current_stock']
    if 'item_sold' not in s.columns and 'units_sold' in s.columns:
        s['item_sold'] = s['units_sold']

    dates = pd.date_range(s.date.min(), s.date.max())
    s['rev'] = s.item_sold * s.price
    d = s.groupby(['date', 'product_id']).agg(q=('item_sold', 'sum'), rev=('rev', 'sum')).reset_index()
    d['price'] = d.rev / d.q
    Q = d.pivot(index='date', columns='product_id', values='q').reindex(dates).fillna(0.0)
    PR = d.pivot(index='date', columns='product_id', values='price').reindex(dates)
    for c in p.product_id:                      # products that never sold
        if c not in Q.columns:
            Q[c] = 0.0
            PR[c] = np.nan
    return p, Q[list(p.product_id)], PR[list(p.product_id)], dates


def active_windows(p, Q, services):
    """(start, end) day index per product. Every product is assumed on sale from day 0.
    Stocked-out products (stock 0, sales stopped) end at their last sale."""
    T = len(Q)
    out = {}
    for _, r in p.iterrows():
        nz = np.nonzero(Q[r.product_id].values)[0]
        if len(nz) == 0:
            out[r.product_id] = (None, None)
        else:
            stocked_out = r.stock_available == 0 and r.product_id not in services
            out[r.product_id] = (0, nz[-1] if stocked_out else T - 1)
    return out


# ----------------------------------------------------------------------------- rate models
def tsb(x, ap=0.1, az=0.1):
    """Teunter-Syntetos-Babai: smoothed P(sale) x smoothed size; decays when sales stop."""
    nz = np.nonzero(x)[0]
    if len(nz) == 0:
        return 0.0
    n0 = min(len(x), 28)
    pr = max((x[:n0] > 0).mean(), 1.0 / n0)
    z = x[nz[0]]
    for v in x:
        if v > 0:
            pr += ap * (1 - pr)
            z += az * (v - z)
        else:
            pr += ap * (0 - pr)
    return pr * z


def moving_avg(x, n):
    return x[-n:].mean()


# ----------------------------------------------------------------------------- GBM
def make_features(Q, PR, catmap, W, t, k, price_override=None):
    """Features for every product at origin day t for target day t+k (uses data up to t only)."""
    rows = []
    maxp = PR.iloc[:t + 1].max()
    tgt_dow = (Q.index[t] + pd.Timedelta(days=k)).dayofweek
    for c in Q.columns:
        st, en = W[c]
        if st is None or st > t:
            continue
        x = Q[c].values[st:min(t, en) + 1]
        if len(x) == 0:
            continue
        pr = PR[c].iloc[:t + 1].dropna()
        last_p = price_override[c] if price_override is not None else (pr.iloc[-1] if len(pr) else np.nan)
        nzi = np.nonzero(x)[0]
        r7, r14, r28 = x[-7:].sum() / 7, x[-14:].mean(), x[-28:].mean()
        rows.append(dict(
            pid=c, k=k, dow=tgt_dow, r7=r7, r14=r14, r28=r28, rall=x.mean(),
            ratio=(r7 + .1) / (r28 + .1),
            ds=min(len(x) - 1 - nzi[-1], 60) if len(nzi) else 60,
            nzm=x[nzi].mean() if len(nzi) else 0,
            nzrate28=(x[-28:] > 0).mean(),
            logp=np.log(last_p) if last_p == last_p else np.nan,
            pratio=last_p / maxp[c] if (last_p == last_p and maxp[c] == maxp[c]) else np.nan,
            cat=catmap[c], age=len(x)))
    return pd.DataFrame(rows)


def build_train(Q, PR, catmap, W, t_last, hmax, tmin=14):
    parts = []
    for t in range(tmin, t_last):
        for k in range(1, min(hmax, t_last - t) + 1):
            f = make_features(Q, PR, catmap, W, t, k)
            if f.empty:
                continue
            f['y'] = [Q[c].values[t + k] if W[c][1] >= t + k else np.nan for c in f.pid]
            parts.append(f)
    return pd.concat(parts).dropna(subset=['y'])


def fit_gbm(df, seed=0):
    m = HistGradientBoostingRegressor(loss='poisson', max_depth=3, learning_rate=0.05, max_iter=150,
                                      min_samples_leaf=80, l2_regularization=5.0, random_state=seed)
    return m.fit(df[FEATS], df.y)


def gbm_totals(models, Q, PR, catmap, W, t, horizon, price_override=None):
    """Sum of predicted daily demand over the next `horizon` days, per product."""
    tot = {}
    for k in range(1, horizon + 1):
        f = make_features(Q, PR, catmap, W, t, k, price_override)
        pred = np.clip(np.mean([m.predict(f[FEATS]) for m in models], axis=0), 0, None)
        for c, v in zip(f.pid, pred):
            tot[c] = tot.get(c, 0.0) + v
    return tot


# ----------------------------------------------------------------------------- back-test
def backtest(p, Q, PR, catmap, W, services, h=14, n_origins=4):
    """Train only on data before each origin, forecast next h days, compare with what happened."""
    T = len(Q)
    rows = []
    for o in [T - 1 - h - 7 * i for i in range(n_origins - 1, -1, -1)]:
        models = [fit_gbm(build_train(Q, PR, catmap, W, o, h))]
        g = gbm_totals(models, Q, PR, catmap, W, o, h)
        for c in Q.columns:
            st, en = W[c]
            if st is None or c in services or en < o + h or (Q[c].values[:o + 1] > 0).sum() < 2:
                continue
            x = Q[c].values[st:o + 1]
            rows.append(dict(origin=o, pid=c, actual=Q[c].values[o + 1:o + h + 1].sum(),
                             MA28=moving_avg(x, 28) * h, TSB=tsb(x) * h, GBM=g.get(c, 0.0)))
    R = pd.DataFrame(rows)
    R['Blend GBM+TSB'] = (R.GBM + R.TSB) / 2
    out = []
    for m in ['MA28', 'TSB', 'GBM', 'Blend GBM+TSB']:
        e = R[m] - R.actual
        out.append((m, e.abs().sum() / R.actual.sum(), e.sum() / R.actual.sum()))
    print(f'\nBack-test: {n_origins} origins, {h}-day horizon, {len(R)} product tests')
    print(pd.DataFrame(out, columns=['model', 'WAPE (error per product)', 'bias (- = under-forecast)'])
          .round(3).to_string(index=False))


# ----------------------------------------------------------------------------- computation
def compute_stock_plan(
    sales,
    products,
    out='stock_plan',
    lead_time=7.0,
    service_level=0.95,
    horizon=30,
    growth=0.03,
    lag=14.0,
    uplift=0.0,
    service_ids='',
    backtest=False,
):
    """
    Computes 30-day demand forecast and replenishment stock plan.
    Accepts CSV paths OR pandas DataFrames for sales and products.
    """
    p, Q, PR, dates = load(sales, products)
    T, H = len(Q), int(horizon)
    services = set(filter(None, service_ids.split(','))) if isinstance(service_ids, str) else (set(service_ids) if hasattr(service_ids, '__iter__') and not isinstance(service_ids, (str, bytes)) else set())
    services |= set(p[p.category.str.contains('service', case=False, na=False)].product_id)   # auto-detect
    catmap = {c: i for c, i in zip(p.product_id, pd.factorize(p.category)[0])}
    W = active_windows(p, Q, services)
    print(f'{len(p)} products, {T} days of history ({dates[0].date()} to {dates[-1].date()}), services: {sorted(services)}')

    if backtest:
        globals()['backtest'](p, Q, PR, catmap, W, services)

    # ---- final models on all data ----
    train = build_train(Q, PR, catmap, W, T - 1, H)
    models = [fit_gbm(train, seed=s) for s in range(5)]
    cur = p.set_index('product_id').current_price
    g30 = gbm_totals(models, Q, PR, catmap, W, T - 1, H, price_override=cur)   # forecast at CURRENT price

    z = norm.ppf(service_level)
    trend = np.exp(np.log(1 + growth) / 7 * (lag + (H + 1) / 2))
    rows = []
    for _, r in p.iterrows():
        c = r.product_id
        st, en = W[c]
        is_service = c in services
        stocked_out = r.stock_available == 0 and not is_service and st is not None
        if st is None:
            base = sigma = 0.0; dsl = None
        else:
            x = Q[c].values[st:en + 1]
            nz = np.nonzero(Q[c].values)[0]
            dsl = T - 1 - nz[-1]
            tsb30 = tsb(x) * H
            dormant = (not stocked_out) and (not is_service) and dsl >= 45      # no sales for 45+ days
            base = (tsb30 if dormant else (tsb30 + g30.get(c, 0.0)) / 2) / H    # units / day
            sigma = float(np.std(x[-56:], ddof=1)) if len(x) > 2 else 0.0

        d = base * trend * (1 + uplift)                 # avg daily demand
        f30 = d * H
        if is_service:
            ss = rop = tgt = order = None; status = 'Service (no stock plan)'
        elif st is None:
            ss = rop = tgt = order = None; status = 'No sales history'
        else:
            ss = int(np.ceil(z * sigma * np.sqrt(lead_time)))
            rop = int(np.ceil(d * lead_time + ss))
            tgt = int(np.ceil(d * (lead_time + H) + ss))
            stock = int(r.stock_available)
            order = max(0, tgt - stock) if stock <= rop else 0
            if f30 < 0.5:      status = 'No recent demand'
            elif stock <= rop: status = 'Understocked'
            elif stock > tgt:  status = 'Overstock'
            else:              status = 'OK'
        rows.append({'Product': r.product_name, 'Avg daily demand': round(d, 1), f'{H}-day forecast': round(f30),
                     'Current stock': 'n/a' if is_service else int(r.stock_available),
                     'Safety stock': ss, 'Reorder point': rop, 'Target stock': tgt, 'Order qty now': order,
                     'Status': status, 'Product ID': c,
                     'Note': ('Stocked out - demand is from before the stock-out' if stocked_out else
                              f'No sales for {dsl} days' if (st is not None and dsl >= 14 and not is_service) else '')})
    out_df = pd.DataFrame(rows)
    for col in ['Safety stock', 'Reorder point', 'Target stock', 'Order qty now']:
        out_df[col] = out_df[col].astype('Int64')          # whole units; blank for services / no history
    rank = {'Understocked': 0, 'OK': 1, 'Overstock': 2, 'No recent demand': 3, 'No sales history': 4, 'Service (no stock plan)': 5}
    out_df['_o'] = out_df.Status.map(rank)
    out_df['_c'] = [(s / d if isinstance(s, int) and d > 0 else 9e9) for s, d in zip(out_df['Current stock'], out_df['Avg daily demand'])]
    out_df = out_df.sort_values(['_o', '_c']).drop(columns=['_o', '_c']).reset_index(drop=True)

    if out:
        out_base = str(Path(out).with_suffix(""))
        out_df.to_csv(out_base + '.csv', index=False)
        try:
            out_df.to_excel(out_base + '.xlsx', index=False, sheet_name='Summary')
        except Exception as exc:
            print(f"[demand] Excel export skipped: {exc}")
        pd.set_option('display.width', 220); pd.set_option('display.max_columns', 20)
        print(f'\nTrend factor {trend:.3f} (growth {growth:.1%}/week), Z={z:.3f}\n')
        print(out_df.drop(columns=['Note']).to_string(index=False))
        print(f'\nWrote {out_base}.csv and {out_base}.xlsx')
    return out_df


def run_from_database(db=None, out=None, **options):
    """
    Executes the demand forecast model using inputs directly from database tables
    `products` and `sales_logs`.
    Saves or updates the output into table `stock_plans` and writes to stock_plan.csv.
    Returns the resulting pandas DataFrame.
    """
    from inventory.db import Product, SalesLog, SessionLocal, save_stock_plan_to_db, ensure_tables

    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True

    try:
        ensure_tables(db.get_bind())

        # 1. Fetch products from database
        prods = db.query(Product).all()
        if not prods:
            raise ValueError("No products found in database table 'products'. Please upload/add products first.")

        p_rows = []
        price_map = {}
        for p in prods:
            unit_p = float(p.unit_price) if p.unit_price is not None else 0.0
            price_map[str(p.product_id).strip()] = unit_p
            p_rows.append({
                "product_id": str(p.product_id).strip(),
                "product_name": str(p.product_name).strip(),
                "current_price": unit_p,
                "category": str(p.category).strip() if p.category else "General",
                "stock_available": int(p.current_stock) if p.current_stock is not None else 0,
            })
        p_df = pd.DataFrame(p_rows).drop_duplicates(subset=['product_id'])

        # 2. Fetch sales logs from database
        sales = db.query(SalesLog).all()
        if not sales:
            raise ValueError("No sales history found in database table 'sales_logs'. Please upload sales history first.")

        s_rows = []
        for s in sales:
            pid = str(s.product_id).strip()
            s_price = float(s.price) if (s.price is not None and s.price > 0) else price_map.get(pid, 0.0)
            s_rows.append({
                "date": s.date,
                "item_sold": int(s.units_sold) if s.units_sold is not None else 0,
                "product_id": pid,
                "price": s_price,
            })
        s_df = pd.DataFrame(s_rows)
        try:
            db.commit()
        except Exception:
            pass

        # Output path defaults to Scripts/stock_plan
        if out is None:
            scripts_dir = Path(__file__).resolve().parent.parent.parent if (Path(__file__).resolve().parent.parent.parent / "sales_history.csv").exists() else Path(__file__).resolve().parent.parent
            out = str(scripts_dir / "stock_plan")

        # 3. Compute stock plan with in-memory DataFrames
        plan_df = compute_stock_plan(sales=s_df, products=p_df, out=out, **options)

        # 4. Save/update to database table stock_plans
        save_stock_plan_to_db(plan_df, db=db)

        return plan_df

    finally:
        if close_session:
            db.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sales', default='sales_history.csv')
    ap.add_argument('--products', default='products.csv')
    ap.add_argument('--out', default='stock_plan', help='output file name without extension (.csv and .xlsx written)')
    ap.add_argument('--lead-time', type=float, default=7, help='days from order to delivery')
    ap.add_argument('--service-level', type=float, default=0.95)
    ap.add_argument('--horizon', type=int, default=30, help='forecast / review period in days')
    ap.add_argument('--growth', type=float, default=0.03, help='assumed weekly demand growth (0.03 = 3%%)')
    ap.add_argument('--lag', type=float, default=14, help='days the growth is applied from before the window')
    ap.add_argument('--uplift', type=float, default=0.0, help='event/festive demand uplift, e.g. 0.2; negative for cautious case')
    ap.add_argument('--service-ids', default='', help='comma list of product ids that are services (no stock)')
    ap.add_argument('--backtest', action='store_true')
    a, _ = ap.parse_known_args(argv)

    return compute_stock_plan(
        sales=a.sales,
        products=a.products,
        out=a.out,
        lead_time=a.lead_time,
        service_level=a.service_level,
        horizon=a.horizon,
        growth=a.growth,
        lag=a.lag,
        uplift=a.uplift,
        service_ids=a.service_ids,
        backtest=a.backtest,
    )


def run(sales='sales_history.csv', products='products.csv', **options):
    """Notebook-friendly entry point. Returns the summary DataFrame."""
    if isinstance(sales, (str, Path)) and isinstance(products, (str, Path)):
        argv = ['--sales', str(sales), '--products', str(products)]
        for k, v in options.items():
            flag = '--' + k.replace('_', '-')
            if v is True:
                argv.append(flag)
            elif v is not False and v is not None:
                argv += [flag, str(v)]
        return main(argv)
    else:
        return compute_stock_plan(sales=sales, products=products, **options)


# ── FastAPI Router ─────────────────────────────────────────────────────────────
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from inventory.db import get_db, StockPlan
from inventory.models.schemas import StockPlanItem, StockPlanResponse

router = APIRouter(tags=["Inventory & Demand Prediction"])


@router.get(
    "/api/demand/stock-plan",
    response_model=StockPlanResponse,
    summary="Compute demand forecast from database and update stock plan",
    description=(
        "Executes demand forecasting (demand.py) taking inputs directly from database tables "
        "'products' and 'sales_logs', saves/updates the output into database table 'stock_plans' "
        "(and writes to stock_plan.csv), and returns the calculated stock plan data."
    ),
)
@router.get(
    "/demand/stock-plan",
    response_model=StockPlanResponse,
    include_in_schema=False,
)
def get_demand_stock_plan_from_db(
    lead_time: float = Query(7.0, description="Days from order to delivery"),
    service_level: float = Query(0.95, description="Target service level (e.g. 0.95)"),
    horizon: int = Query(30, description="Forecast review period in days"),
    growth: float = Query(0.03, description="Assumed weekly demand growth (0.03 = 3%)"),
    lag: float = Query(14.0, description="Days growth applied from before window"),
    uplift: float = Query(0.0, description="Demand uplift factor (e.g. 0.0)"),
    service_ids: str = Query("", description="Comma-separated product IDs that are services"),
    status: Optional[str] = Query(None, description="Optional filter by status (Understocked, OK, Overstock, etc.)"),
    db: Session = Depends(get_db),
):
    """
    Connects to demand.py, reads inputs from database tables 'products' and 'sales_logs',
    runs the gradient-boosting + TSB demand forecasting model, updates table 'stock_plans',
    and returns the resulting stock plan.
    """
    from fastapi.params import Query as QueryParam
    _lead_time = lead_time.default if isinstance(lead_time, QueryParam) else lead_time
    _service_level = service_level.default if isinstance(service_level, QueryParam) else service_level
    _horizon = horizon.default if isinstance(horizon, QueryParam) else horizon
    _growth = growth.default if isinstance(growth, QueryParam) else growth
    _lag = lag.default if isinstance(lag, QueryParam) else lag
    _uplift = uplift.default if isinstance(uplift, QueryParam) else uplift
    _service_ids = service_ids.default if isinstance(service_ids, QueryParam) else service_ids
    _status = status.default if isinstance(status, QueryParam) else status

    try:
        run_from_database(
            db=db,
            lead_time=_lead_time,
            service_level=_service_level,
            horizon=_horizon,
            growth=_growth,
            lag=_lag,
            uplift=_uplift,
            service_ids=_service_ids,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing demand model from database: {exc}",
        )

    query = db.query(StockPlan)
    if _status:
        query = query.filter(StockPlan.status.ilike(f"%{_status.strip()}%"))
    records = query.all()

    items = [
        StockPlanItem(
            product_id=r.product_id,
            product_name=r.product_name,
            avg_daily_demand=r.avg_daily_demand,
            forecast_30d=r.forecast_30d,
            current_stock=r.current_stock,
            safety_stock=r.safety_stock,
            reorder_point=r.reorder_point,
            target_stock=r.target_stock,
            order_qty_now=r.order_qty_now,
            status=r.status,
            note=r.note,
        )
        for r in records
    ]

    understocked = sum(1 for i in items if i.status == "Understocked")
    overstock = sum(1 for i in items if i.status == "Overstock")
    ok_count = sum(1 for i in items if i.status == "OK")

    scripts_dir = Path(__file__).resolve().parent.parent.parent if (Path(__file__).resolve().parent.parent.parent / "sales_history.csv").exists() else Path(__file__).resolve().parent.parent
    csv_file = str(scripts_dir / "stock_plan.csv")

    return StockPlanResponse(
        status="success",
        total_items=len(items),
        understocked_count=understocked,
        overstock_count=overstock,
        ok_count=ok_count,
        csv_path=csv_file,
        items=items,
        data=items,
    )


if __name__ == '__main__':
    main()
