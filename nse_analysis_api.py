"""
NSE Stock Analysis API v2 — for n8n / Vercel frontend integration
================================================================================
Everything from v1 (/analyze, /chart, /watchlist) is kept and backward compatible.

NEW IN v2
---------
  1. In-memory caching (5-15 min)     -> far fewer Yahoo rate-limit errors on Render
  2. More indicators                  -> ATR, Stochastic, OBV trend, volume spike,
                                         pivot points, support/resistance
  3. Composite signal score           -> technical + fundamental score and a rating
  4. /market                          -> Nifty, Bank Nifty, Sensex, Nifty IT, India VIX
  5. /compare                         -> rebased performance + key stats for 2-5 stocks
  6. /risk/{symbol}                   -> volatility, max drawdown, Sharpe, beta vs Nifty
  7. /backtest/{symbol}               -> MA-crossover or RSI strategy vs buy & hold
  8. /screener                        -> scan Nifty 50 for oversold, golden cross, etc.
  9. /news/{symbol}                   -> latest headlines (best effort)
 10. Parallel /watchlist              -> faster, up to 20 symbols
 11. Optional API-key protection      -> set API_KEY env var on Render

RUN LOCALLY
-----------
    pip install -r requirements.txt
    uvicorn nse_analysis_api:app --reload --port 8000
    open http://127.0.0.1:8000/docs   (interactive docs for every endpoint)

NOT INVESTMENT ADVICE: scores and signals are rule-based indicators only.
"""

import os
import time
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

API_KEY = os.getenv("API_KEY")  # leave unset = open API; set it on Render to protect it
RISK_FREE_RATE = 0.065          # approx. Indian 10y G-sec yield, used for Sharpe

app = FastAPI(title="NSE Stock Analysis API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Vercel URL later, e.g. ["https://my-terminal.vercel.app"]
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_key(x_api_key: str = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


# ── CACHE ─────────────────────────────────────────────────────────────────

_cache = {}
_cache_lock = threading.Lock()


def cached(key, ttl_seconds, producer):
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl_seconds:
            return hit[1]
    value = producer()
    with _cache_lock:
        _cache[key] = (now, value)
        if len(_cache) > 500:  # crude memory guard for free-tier hosting
            oldest = sorted(_cache.items(), key=lambda kv: kv[1][0])[:100]
            for k, _ in oldest:
                _cache.pop(k, None)
    return value


def to_ticker(symbol):
    symbol = symbol.upper().strip()
    if symbol.startswith("^") or symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    return f"{symbol}.NS"


def get_history(ticker, period="2y", ttl=300):
    def fetch():
        return yf.Ticker(ticker).history(period=period, auto_adjust=True)

    hist = cached(f"hist:{ticker}:{period}", ttl, fetch)
    if hist is None or hist.empty:
        with _cache_lock:
            _cache.pop(f"hist:{ticker}:{period}", None)  # do not cache failures
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}. Check the symbol is correct.")
    return hist


# ── INDICATORS (v1) ───────────────────────────────────────────────────────

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_roc(prices, period=12):
    return prices.pct_change(period) * 100


def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def calculate_moving_averages(prices, short_window=50, long_window=200):
    return prices.rolling(short_window).mean(), prices.rolling(long_window).mean()


def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0; minus_dm[(minus_dm - plus_dm) < 0] = 0
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def calculate_52w_high_proximity(prices, window=252):
    return (prices / prices.rolling(window).max()) * 100


def calculate_bollinger_bands(prices, window=20, num_std=2):
    middle = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    percent_b = (prices - lower) / (upper - lower)
    bandwidth = (upper - lower) / middle * 100
    return percent_b, bandwidth


# ── INDICATORS (NEW in v2) ────────────────────────────────────────────────

def calculate_atr(high, low, close, period=14):
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest)
    return k, k.rolling(d_period).mean()


def calculate_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def calculate_pivots(high, low, close):
    """Classic floor-trader pivots from the last completed session."""
    h, l, c = float(high.iloc[-1]), float(low.iloc[-1]), float(close.iloc[-1])
    p = (h + l + c) / 3
    return {
        "pivot": safe_round(p),
        "r1": safe_round(2 * p - l), "r2": safe_round(p + (h - l)),
        "s1": safe_round(2 * p - h), "s2": safe_round(p - (h - l)),
    }


# ── FUNDAMENTALS (v1, unchanged logic) ────────────────────────────────────

def safe_round(val, decimals=2):
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return None


def get_yoy_earnings_growth(net_income, tolerance_days=45):
    """Date-matched YoY growth (fixes irregularly reported quarters)."""
    if len(net_income) < 2:
        return None, None
    latest_date = net_income.index[0]
    latest_val = net_income.iloc[0]
    target_date = latest_date - pd.DateOffset(months=12)
    date_diffs = (net_income.index - target_date).days.to_series(index=net_income.index).abs()
    closest_idx = date_diffs.idxmin()
    if date_diffs[closest_idx] > tolerance_days:
        return None, None
    base_val = net_income.loc[closest_idx]
    if base_val == 0 or pd.isna(base_val):
        return None, None
    return round(((latest_val / base_val) - 1) * 100, 2), str(closest_idx.date())


def get_yoy_revenue_growth(revenue, tolerance_days=45):
    return get_yoy_earnings_growth(revenue, tolerance_days)


def get_fundamentals(stock):
    """Built from income statement + balance sheet only (avoids blocked stock.info)."""
    fundamentals = {
        "yoy_earnings_growth_pct": None, "earnings_base_quarter": None,
        "yoy_revenue_growth_pct": None, "revenue_base_quarter": None,
        "operating_margin_pct": None, "net_margin_pct": None,
        "roe_pct": None, "debt_to_equity": None, "market_cap_cr": None,
    }
    latest_revenue = latest_net_income = latest_operating_income = None

    try:
        income_stmt = stock.quarterly_income_stmt
        if income_stmt is not None:
            if "Net Income" in income_stmt.index:
                net_income = income_stmt.loc["Net Income"].dropna()
                growth, base_date = get_yoy_earnings_growth(net_income)
                fundamentals["yoy_earnings_growth_pct"] = growth
                fundamentals["earnings_base_quarter"] = base_date
                if len(net_income) > 0:
                    latest_net_income = net_income.iloc[0]

            if "Total Revenue" in income_stmt.index:
                revenue = income_stmt.loc["Total Revenue"].dropna()
                growth, base_date = get_yoy_revenue_growth(revenue)
                fundamentals["yoy_revenue_growth_pct"] = growth
                fundamentals["revenue_base_quarter"] = base_date
                if len(revenue) > 0:
                    latest_revenue = revenue.iloc[0]

            if "Operating Income" in income_stmt.index:
                op_income = income_stmt.loc["Operating Income"].dropna()
                if len(op_income) > 0:
                    latest_operating_income = op_income.iloc[0]

        if latest_revenue and latest_revenue != 0:
            if latest_net_income is not None:
                fundamentals["net_margin_pct"] = safe_round((latest_net_income / latest_revenue) * 100)
            if latest_operating_income is not None:
                fundamentals["operating_margin_pct"] = safe_round((latest_operating_income / latest_revenue) * 100)
    except Exception as e:
        fundamentals["_debug_income_stmt_error"] = str(e)

    try:
        balance_sheet = stock.quarterly_balance_sheet
        if balance_sheet is not None:
            equity = total_debt = None
            for label in ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                if label in balance_sheet.index:
                    s = balance_sheet.loc[label].dropna()
                    if len(s) > 0:
                        equity = s.iloc[0]
                        break
            for label in ["Total Debt", "Net Debt"]:
                if label in balance_sheet.index:
                    s = balance_sheet.loc[label].dropna()
                    if len(s) > 0:
                        total_debt = s.iloc[0]
                        break
            if equity and equity != 0:
                if total_debt is not None:
                    fundamentals["debt_to_equity"] = safe_round(total_debt / equity)
                if latest_net_income is not None:
                    fundamentals["roe_pct"] = safe_round(((latest_net_income * 4) / equity) * 100)
    except Exception as e:
        fundamentals["_debug_balance_sheet_error"] = str(e)

    try:
        fast_info = stock.fast_info
        if fast_info and fast_info.get("marketCap"):
            fundamentals["market_cap_cr"] = safe_round(fast_info["marketCap"] / 1e7)
    except Exception:
        pass

    return fundamentals


# ── TECHNICAL SNAPSHOT ────────────────────────────────────────────────────

def get_technical_snapshot(hist):
    close, high, low, volume = hist["Close"], hist["High"], hist["Low"], hist["Volume"]

    corporate_action_flag = bool((close.pct_change().abs() > 0.15).any())

    rsi = calculate_rsi(close)
    roc = calculate_roc(close)
    macd_line, signal_line, macd_hist = calculate_macd(close)
    ma50, ma200 = calculate_moving_averages(close)
    adx = calculate_adx(high, low, close)
    high52 = calculate_52w_high_proximity(close)
    percent_b, bandwidth = calculate_bollinger_bands(close)
    atr = calculate_atr(high, low, close)
    stoch_k, stoch_d = calculate_stochastic(high, low, close)
    obv = calculate_obv(close, volume)
    avg_vol20 = volume.rolling(20).mean()

    ma_trend = None
    if not pd.isna(ma50.iloc[-1]) and not pd.isna(ma200.iloc[-1]):
        ma_trend = "bullish" if ma50.iloc[-1] > ma200.iloc[-1] else "bearish"

    # golden / death cross in the last 10 sessions
    cross_event = None
    diff = (ma50 - ma200).dropna().tail(11)
    if len(diff) > 1:
        if diff.iloc[-1] > 0 and (diff.iloc[:-1] <= 0).any():
            cross_event = "golden_cross"
        elif diff.iloc[-1] < 0 and (diff.iloc[:-1] >= 0).any():
            cross_event = "death_cross"

    last_close = float(close.iloc[-1])
    last_atr = atr.iloc[-1]
    vol_ratio = (volume.iloc[-1] / avg_vol20.iloc[-1]) if avg_vol20.iloc[-1] else None
    obv_trend = None
    obv_ma = obv.rolling(20).mean().iloc[-1]
    if not pd.isna(obv_ma):
        obv_trend = "accumulation" if obv.iloc[-1] > obv_ma else "distribution"

    technical = {
        "current_price": safe_round(last_close),
        "day_change_pct": safe_round(close.pct_change().iloc[-1] * 100),
        "rsi_14": safe_round(rsi.iloc[-1]),
        "roc_12": safe_round(roc.iloc[-1]),
        "macd_line": safe_round(macd_line.iloc[-1]),
        "macd_signal": safe_round(signal_line.iloc[-1]),
        "macd_histogram": safe_round(macd_hist.iloc[-1]),
        "ma_50": safe_round(ma50.iloc[-1]),
        "ma_200": safe_round(ma200.iloc[-1]),
        "ma_trend": ma_trend,
        "ma_cross_event": cross_event,
        "adx_14": safe_round(adx.iloc[-1]),
        "pct_of_52w_high": safe_round(high52.iloc[-1]),
        "week52_high": safe_round(close.tail(252).max()),
        "week52_low": safe_round(close.tail(252).min()),
        "bollinger_percent_b": safe_round(percent_b.iloc[-1], 3),
        "bollinger_bandwidth_pct": safe_round(bandwidth.iloc[-1]),
        # new in v2
        "atr_14": safe_round(last_atr),
        "atr_pct": safe_round(last_atr / last_close * 100) if last_close else None,
        "stochastic_k": safe_round(stoch_k.iloc[-1]),
        "stochastic_d": safe_round(stoch_d.iloc[-1]),
        "obv_trend": obv_trend,
        "volume_ratio_20d": safe_round(vol_ratio),
        "volume_spike": bool(vol_ratio is not None and vol_ratio >= 2),
    }

    levels = {
        **calculate_pivots(high, low, close),
        "support_20d": safe_round(low.tail(20).min()),
        "resistance_20d": safe_round(high.tail(20).max()),
        # ATR-based stop suggestion (2x ATR below price) — a risk-sizing aid, not a recommendation
        "atr_stop_2x": safe_round(last_close - 2 * last_atr) if not pd.isna(last_atr) else None,
    }
    return technical, levels, corporate_action_flag


# ── SIGNAL SCORING (NEW) ──────────────────────────────────────────────────

def rating_from_score(score):
    if score is None:
        return "Insufficient data"
    if score >= 60: return "Strong Bullish"
    if score >= 25: return "Bullish"
    if score > -25: return "Neutral"
    if score > -60: return "Bearish"
    return "Strong Bearish"


def score_technical(t):
    pts, max_pts, reasons = 0, 0, []

    def add(cond_bull, cond_bear, weight, bull_msg, bear_msg):
        nonlocal pts, max_pts
        max_pts += weight
        if cond_bull:
            pts += weight; reasons.append(bull_msg)
        elif cond_bear:
            pts -= weight; reasons.append(bear_msg)

    rsi = t["rsi_14"]
    if rsi is not None:
        add(rsi < 30, rsi > 70, 2, f"RSI {rsi} oversold (bounce potential)", f"RSI {rsi} overbought")
    if t["macd_histogram"] is not None:
        add(t["macd_histogram"] > 0, t["macd_histogram"] <= 0, 2, "MACD histogram positive", "MACD histogram negative")
    if t["ma_trend"]:
        add(t["ma_trend"] == "bullish", t["ma_trend"] == "bearish", 2, "50-DMA above 200-DMA", "50-DMA below 200-DMA")
    if t["ma_50"] and t["current_price"]:
        add(t["current_price"] > t["ma_50"], t["current_price"] <= t["ma_50"], 1, "Price above 50-DMA", "Price below 50-DMA")
    if t["stochastic_k"] is not None:
        add(t["stochastic_k"] < 20, t["stochastic_k"] > 80, 1, "Stochastic oversold", "Stochastic overbought")
    if t["obv_trend"]:
        add(t["obv_trend"] == "accumulation", t["obv_trend"] == "distribution", 1,
            "OBV shows accumulation", "OBV shows distribution")
    if t["bollinger_percent_b"] is not None:
        add(t["bollinger_percent_b"] < 0.05, t["bollinger_percent_b"] > 0.95, 1,
            "Price at lower Bollinger band", "Price at upper Bollinger band")

    if t["ma_cross_event"] == "golden_cross":
        pts += 2; max_pts += 2; reasons.append("Fresh golden cross (last 10 sessions)")
    elif t["ma_cross_event"] == "death_cross":
        pts -= 2; max_pts += 2; reasons.append("Fresh death cross (last 10 sessions)")

    score = round(pts / max_pts * 100) if max_pts else None
    note = None
    if t["adx_14"] is not None:
        note = "Strong trend (ADX>25): trend signals more reliable" if t["adx_14"] > 25 \
            else "Weak trend (ADX<20): trend signals less reliable" if t["adx_14"] < 20 else None
    return {"score": score, "rating": rating_from_score(score), "reasons": reasons, "trend_strength_note": note}


def score_fundamental(f):
    pts, max_pts, reasons = 0, 0, []

    def add(val, good, bad, good_msg, bad_msg):
        nonlocal pts, max_pts
        if val is None:
            return
        max_pts += 1
        if good(val):
            pts += 1; reasons.append(good_msg)
        elif bad(val):
            pts -= 1; reasons.append(bad_msg)

    add(f["yoy_earnings_growth_pct"], lambda v: v > 15, lambda v: v < 0, "Earnings growing >15% YoY", "Earnings declining YoY")
    add(f["yoy_revenue_growth_pct"], lambda v: v > 10, lambda v: v < 0, "Revenue growing >10% YoY", "Revenue declining YoY")
    add(f["net_margin_pct"], lambda v: v > 10, lambda v: v < 3, "Healthy net margin (>10%)", "Thin net margin (<3%)")
    add(f["roe_pct"], lambda v: v > 15, lambda v: v < 8, "ROE above 15%", "ROE below 8%")
    add(f["debt_to_equity"], lambda v: v < 1, lambda v: v > 2, "Low leverage (D/E<1)", "High leverage (D/E>2)")

    score = round(pts / max_pts * 100) if max_pts else None
    return {"score": score, "rating": rating_from_score(score), "reasons": reasons}


def combine_scores(tech, fund):
    """60% technical / 40% fundamental; falls back to whichever is available."""
    ts, fs = tech["score"], fund["score"]
    if ts is None and fs is None:
        return None
    if ts is None: return fs
    if fs is None: return ts
    return round(ts * 0.6 + fs * 0.4)


# ── ENDPOINTS ─────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "ok", "message": "NSE Stock Analysis API v2 is running",
            "docs": "/docs", "auth_required": bool(API_KEY)}


@app.get("/analyze/{symbol}", dependencies=[Depends(verify_key)])
def analyze_stock(symbol: str):
    """Full technical + fundamental analysis with levels and composite signal."""
    ticker = to_ticker(symbol)
    try:
        hist = get_history(ticker)
        stock = yf.Ticker(ticker)

        technical, levels, corporate_action_flag = get_technical_snapshot(hist)
        fundamentals = cached(f"fund:{ticker}", 6 * 3600, lambda: get_fundamentals(stock))

        tech_signal = score_technical(technical)
        fund_signal = score_fundamental(fundamentals)
        overall = combine_scores(tech_signal, fund_signal)

        return {
            "symbol": ticker,
            "company_name": ticker,
            "timestamp": datetime.now().isoformat(),
            "corporate_action_flag": corporate_action_flag,
            "corporate_action_note": (
                "A >15% single-day price move was detected in the last 2 years — "
                "moving-average and Bollinger figures may be distorted by a demerger, "
                "split, or similar event. Verify before trusting the technical signals."
                if corporate_action_flag else None
            ),
            "technical": technical,
            "levels": levels,
            "fundamentals": fundamentals,
            "signal": {
                "overall_score": overall,
                "overall_rating": rating_from_score(overall),
                "technical": tech_signal,
                "fundamental": fund_signal,
                "disclaimer": "Rule-based indicator summary, not investment advice.",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing {ticker}: {str(e)}")


@app.get("/chart/{symbol}", dependencies=[Depends(verify_key)])
def chart_data(symbol: str, months: int = 6):
    """OHLCV + RSI, MACD, MA50/200, Bollinger bands, stochastic for charting."""
    ticker = to_ticker(symbol)
    months = min(max(months, 1), 24)
    try:
        hist = get_history(ticker)
        close = hist["Close"]
        rsi = calculate_rsi(close)
        macd_line, signal_line, macd_hist = calculate_macd(close)
        ma50, ma200 = calculate_moving_averages(close)
        mid = close.rolling(20).mean()
        std = close.rolling(20).std()
        bb_up, bb_low = mid + 2 * std, mid - 2 * std
        stoch_k, stoch_d = calculate_stochastic(hist["High"], hist["Low"], close)

        recent = hist.tail(months * 21)
        rows = []
        for date, row in recent.iterrows():
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": safe_round(row["Open"]), "high": safe_round(row["High"]),
                "low": safe_round(row["Low"]), "close": safe_round(row["Close"]),
                "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else None,
                "rsi": safe_round(rsi.loc[date]),
                "macd_line": safe_round(macd_line.loc[date]),
                "macd_signal": safe_round(signal_line.loc[date]),
                "macd_histogram": safe_round(macd_hist.loc[date]),
                "ma_50": safe_round(ma50.loc[date]), "ma_200": safe_round(ma200.loc[date]),
                "bb_upper": safe_round(bb_up.loc[date]), "bb_middle": safe_round(mid.loc[date]),
                "bb_lower": safe_round(bb_low.loc[date]),
                "stoch_k": safe_round(stoch_k.loc[date]), "stoch_d": safe_round(stoch_d.loc[date]),
            })
        return {"symbol": ticker, "months": months, "data": rows}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chart data for {ticker}: {str(e)}")


def _watch_row(symbol):
    ticker = to_ticker(symbol)
    try:
        hist = get_history(ticker)
        technical, _, flag = get_technical_snapshot(hist)
        sig = score_technical(technical)
        return {
            "symbol": ticker, "company_name": ticker,
            "current_price": technical["current_price"],
            "day_change_pct": technical["day_change_pct"],
            "rsi_14": technical["rsi_14"],
            "macd_histogram": technical["macd_histogram"],
            "ma_trend": technical["ma_trend"],
            "pct_of_52w_high": technical["pct_of_52w_high"],
            "volume_spike": technical["volume_spike"],
            "technical_score": sig["score"],
            "technical_rating": sig["rating"],
            "corporate_action_flag": flag,
        }
    except HTTPException as e:
        return {"symbol": ticker, "error": e.detail}
    except Exception as e:
        return {"symbol": ticker, "error": str(e)}


@app.get("/watchlist", dependencies=[Depends(verify_key)])
def watchlist(symbols: str):
    """/watchlist?symbols=RELIANCE,TCS,INFY  (max 20, fetched in parallel)"""
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()][:20]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="Provide at least one symbol, e.g. ?symbols=RELIANCE,TCS")
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(_watch_row, symbol_list))
    return {"count": len(results), "stocks": results}


# ── NEW: MARKET OVERVIEW ──────────────────────────────────────────────────

INDICES = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "NIFTY IT": "^CNXIT",
    "INDIA VIX": "^INDIAVIX",
}


def _index_row(item):
    name, ticker = item
    try:
        hist = get_history(ticker, period="1y", ttl=120)
        close = hist["Close"]
        ma200 = close.rolling(200).mean().iloc[-1]
        return {
            "name": name, "ticker": ticker,
            "last": safe_round(close.iloc[-1]),
            "change_pct": safe_round(close.pct_change().iloc[-1] * 100),
            "rsi_14": safe_round(calculate_rsi(close).iloc[-1]),
            "above_200dma": None if pd.isna(ma200) else bool(close.iloc[-1] > ma200),
            "pct_of_52w_high": safe_round(calculate_52w_high_proximity(close).iloc[-1]),
        }
    except Exception as e:
        return {"name": name, "ticker": ticker, "error": str(getattr(e, "detail", e))}


@app.get("/market", dependencies=[Depends(verify_key)])
def market_overview():
    """Index dashboard: Nifty, Bank Nifty, Sensex, Nifty IT, India VIX + a simple mood read."""
    with ThreadPoolExecutor(max_workers=5) as pool:
        rows = list(pool.map(_index_row, INDICES.items()))
    by_name = {r["name"]: r for r in rows}

    mood = None
    nifty, vix = by_name.get("NIFTY 50", {}), by_name.get("INDIA VIX", {})
    if nifty.get("above_200dma") is not None and vix.get("last") is not None:
        if nifty["above_200dma"] and vix["last"] < 16:
            mood = "Calm uptrend (Nifty above 200-DMA, low VIX)"
        elif nifty["above_200dma"]:
            mood = "Uptrend with elevated volatility"
        elif vix["last"] > 20:
            mood = "Stressed (Nifty below 200-DMA, high VIX)"
        else:
            mood = "Weak trend (Nifty below 200-DMA)"
    return {"timestamp": datetime.now().isoformat(), "market_mood": mood, "indices": rows}


# ── NEW: COMPARE ──────────────────────────────────────────────────────────

@app.get("/compare", dependencies=[Depends(verify_key)])
def compare(symbols: str, months: int = 12):
    """/compare?symbols=TCS,INFY,WIPRO&months=12 — performance rebased to 100 + key stats."""
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()][:5]
    if len(symbol_list) < 2:
        raise HTTPException(status_code=400, detail="Provide 2-5 symbols, e.g. ?symbols=TCS,INFY")
    months = min(max(months, 1), 24)

    series, stats = {}, []
    for sym in symbol_list:
        ticker = to_ticker(sym)
        try:
            hist = get_history(ticker)
            technical, _, _ = get_technical_snapshot(hist)
            sig = score_technical(technical)
            close = hist["Close"].tail(months * 21)
            series[ticker] = close
            stats.append({
                "symbol": ticker,
                "return_pct": safe_round((close.iloc[-1] / close.iloc[0] - 1) * 100),
                "rsi_14": technical["rsi_14"], "ma_trend": technical["ma_trend"],
                "pct_of_52w_high": technical["pct_of_52w_high"],
                "technical_score": sig["score"], "technical_rating": sig["rating"],
            })
        except HTTPException as e:
            stats.append({"symbol": ticker, "error": e.detail})

    if not series:
        raise HTTPException(status_code=404, detail="No valid symbols found.")
    df = pd.DataFrame({k: v.rename(lambda d: d.strftime("%Y-%m-%d")) for k, v in series.items()}).dropna()
    rebased = (df / df.iloc[0] * 100).round(2)
    rows = [{"date": d, **{k: float(v) for k, v in r.items()}} for d, r in rebased.iterrows()]
    best = max((s for s in stats if "return_pct" in s), key=lambda s: s["return_pct"], default=None)
    return {"months": months, "best_performer": best["symbol"] if best else None,
            "stats": stats, "rebased_series": rows}


# ── NEW: RISK METRICS ─────────────────────────────────────────────────────

def _daily_returns(hist):
    r = hist["Close"].pct_change().dropna()
    r.index = r.index.tz_localize(None).normalize() if r.index.tz is not None else r.index.normalize()
    return r


@app.get("/risk/{symbol}", dependencies=[Depends(verify_key)])
def risk_metrics(symbol: str):
    """Volatility, max drawdown, Sharpe, Sortino, beta vs Nifty 50, 95% 1-day VaR (1y data)."""
    ticker = to_ticker(symbol)
    try:
        hist = get_history(ticker).tail(252)
        r = _daily_returns(hist)
        ann_vol = r.std() * np.sqrt(252)
        ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
        equity = (1 + r).cumprod()
        drawdown = equity / equity.cummax() - 1
        downside = r[r < 0].std() * np.sqrt(252)

        beta = correlation = None
        try:
            nifty_r = _daily_returns(get_history("^NSEI"))
            joined = pd.concat([r, nifty_r], axis=1, join="inner").dropna().tail(252)
            joined.columns = ["s", "n"]
            if len(joined) > 60:
                beta = safe_round(joined["s"].cov(joined["n"]) / joined["n"].var())
                correlation = safe_round(joined["s"].corr(joined["n"]))
        except Exception:
            pass

        risk_level = None
        if not pd.isna(ann_vol):
            risk_level = "Low" if ann_vol < 0.20 else "Moderate" if ann_vol < 0.35 else "High"

        return {
            "symbol": ticker, "period": "last ~1 year",
            "annualised_return_pct": safe_round(ann_ret * 100),
            "annualised_volatility_pct": safe_round(ann_vol * 100),
            "sharpe_ratio": safe_round((ann_ret - RISK_FREE_RATE) / ann_vol) if ann_vol else None,
            "sortino_ratio": safe_round((ann_ret - RISK_FREE_RATE) / downside) if downside else None,
            "max_drawdown_pct": safe_round(drawdown.min() * 100),
            "var_95_1day_pct": safe_round(np.percentile(r, 5) * 100),
            "beta_vs_nifty": beta, "correlation_vs_nifty": correlation,
            "risk_level": risk_level,
            "risk_free_rate_assumed_pct": RISK_FREE_RATE * 100,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing risk for {ticker}: {str(e)}")


# ── NEW: BACKTEST ─────────────────────────────────────────────────────────

@app.get("/backtest/{symbol}", dependencies=[Depends(verify_key)])
def backtest(symbol: str, strategy: str = "ma_cross", fast: int = 20, slow: int = 50, years: int = 5):
    """
    strategy=ma_cross : long when SMA(fast) > SMA(slow), else cash
    strategy=rsi      : buy when RSI<30, sell when RSI>55
    Signals execute on the NEXT day's close (no look-ahead). No costs/taxes/slippage modelled.
    """
    ticker = to_ticker(symbol)
    years = min(max(years, 1), 10)
    if strategy not in ("ma_cross", "rsi"):
        raise HTTPException(status_code=400, detail="strategy must be 'ma_cross' or 'rsi'")
    if strategy == "ma_cross" and not (2 <= fast < slow <= 250):
        raise HTTPException(status_code=400, detail="Need 2 <= fast < slow <= 250")
    try:
        close = get_history(ticker, period=f"{years}y", ttl=900)["Close"]

        if strategy == "ma_cross":
            signal = (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(int)
            signal[close.rolling(slow).mean().isna()] = 0
        else:
            rsi = calculate_rsi(close)
            state, vals = 0, []
            for v in rsi:
                if not pd.isna(v):
                    if state == 0 and v < 30: state = 1
                    elif state == 1 and v > 55: state = 0
                vals.append(state)
            signal = pd.Series(vals, index=close.index)

        pos = signal.shift(1).fillna(0)
        ret = close.pct_change().fillna(0)
        strat_ret = pos * ret
        equity, bh = (1 + strat_ret).cumprod(), (1 + ret).cumprod()

        entries = (pos.diff() == 1) | ((pos == 1) & (pos.index == pos.index[0]))
        trade_id = entries.cumsum() * (pos == 1)
        trade_returns = (1 + strat_ret[trade_id > 0]).groupby(trade_id[trade_id > 0]).prod() - 1
        n_years = len(close) / 252

        def cagr(curve): return (curve.iloc[-1] ** (1 / n_years) - 1) * 100 if n_years > 0 else None
        def mdd(curve): return ((curve / curve.cummax()) - 1).min() * 100

        return {
            "symbol": ticker, "strategy_name": strategy,
            "params": {"fast": fast, "slow": slow} if strategy == "ma_cross" else {"buy_below_rsi": 30, "sell_above_rsi": 55},
            "years_tested": round(n_years, 1),
            "strategy": {
                "name": strategy,
                "total_return_pct": safe_round((equity.iloc[-1] - 1) * 100),
                "cagr_pct": safe_round(cagr(equity)),
                "max_drawdown_pct": safe_round(mdd(equity)),
                "trades": int(len(trade_returns)),
                "win_rate_pct": safe_round((trade_returns > 0).mean() * 100) if len(trade_returns) else None,
                "avg_trade_pct": safe_round(trade_returns.mean() * 100) if len(trade_returns) else None,
                "time_in_market_pct": safe_round(pos.mean() * 100),
            },
            "buy_and_hold": {
                "total_return_pct": safe_round((bh.iloc[-1] - 1) * 100),
                "cagr_pct": safe_round(cagr(bh)),
                "max_drawdown_pct": safe_round(mdd(bh)),
            },
            "currently_in_position": bool(signal.iloc[-1] == 1),
            "equity_curve": [
                {"date": d.strftime("%Y-%m-%d"), "strategy": round(float(equity.loc[d]), 4), "buy_hold": round(float(bh.loc[d]), 4)}
                for d in equity.index[::5]  # every 5th day keeps the payload small
            ],
            "caveat": "Past performance does not predict future results. Costs, taxes and slippage are not modelled.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed for {ticker}: {str(e)}")


# ── NEW: SCREENER ─────────────────────────────────────────────────────────

# Edit this list as the index changes (constituents are revised periodically).
NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE",
    "BAJAJFINSV", "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK",
    "INDUSINDBK", "INFY", "ITC", "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND",
    "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA",
    "TATACONSUMER", "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]

SCREEN_FILTERS = {
    "oversold": lambda t: t["rsi_14"] is not None and t["rsi_14"] < 30,
    "overbought": lambda t: t["rsi_14"] is not None and t["rsi_14"] > 70,
    "golden_cross": lambda t: t["ma_cross_event"] == "golden_cross",
    "death_cross": lambda t: t["ma_cross_event"] == "death_cross",
    "near_52w_high": lambda t: t["pct_of_52w_high"] is not None and t["pct_of_52w_high"] >= 95,
    "volume_spike": lambda t: t["volume_spike"],
    "strong_trend": lambda t: t["adx_14"] is not None and t["adx_14"] > 25 and t["ma_trend"] == "bullish",
    "bullish_macd": lambda t: t["macd_histogram"] is not None and t["macd_histogram"] > 0 and t["rsi_14"] is not None and t["rsi_14"] < 65,
}


def _scan_one(symbol):
    ticker = to_ticker(symbol)
    try:
        hist = get_history(ticker, ttl=900)
        t, _, flag = get_technical_snapshot(hist)
        sig = score_technical(t)
        return {**t, "symbol": ticker, "technical_score": sig["score"],
                "technical_rating": sig["rating"], "corporate_action_flag": flag}
    except Exception:
        return None


@app.get("/screener", dependencies=[Depends(verify_key)])
def screener(filter: str = "oversold", symbols: str = None):
    """
    /screener?filter=oversold                      -> scans Nifty 50
    /screener?filter=golden_cross&symbols=TCS,INFY -> scans your own list (max 60)
    Filters: oversold, overbought, golden_cross, death_cross, near_52w_high,
             volume_spike, strong_trend, bullish_macd
    First call can take 20-40s (50 stocks); later calls are cached for 15 min.
    """
    if filter not in SCREEN_FILTERS:
        raise HTTPException(status_code=400, detail=f"filter must be one of: {', '.join(SCREEN_FILTERS)}")
    universe = [s.strip() for s in symbols.split(",") if s.strip()][:60] if symbols else NIFTY50

    with ThreadPoolExecutor(max_workers=6) as pool:
        scanned = [r for r in pool.map(_scan_one, universe) if r]

    matches = [r for r in scanned if SCREEN_FILTERS[filter](r)]
    matches.sort(key=lambda r: r["technical_score"] if r["technical_score"] is not None else -999, reverse=True)
    keep = ["symbol", "current_price", "day_change_pct", "rsi_14", "macd_histogram", "ma_trend",
            "adx_14", "pct_of_52w_high", "volume_ratio_20d", "technical_score", "technical_rating",
            "corporate_action_flag"]
    return {
        "filter": filter, "universe_size": len(universe), "scanned_ok": len(scanned),
        "match_count": len(matches),
        "matches": [{k: m[k] for k in keep} for m in matches],
    }


# ── NEW: NEWS ─────────────────────────────────────────────────────────────

@app.get("/news/{symbol}", dependencies=[Depends(verify_key)])
def news(symbol: str, limit: int = 8):
    """Latest headlines (best effort — Yahoo's news feed can be empty or blocked on cloud hosts)."""
    ticker = to_ticker(symbol)
    limit = min(max(limit, 1), 20)
    try:
        raw = cached(f"news:{ticker}", 900, lambda: yf.Ticker(ticker).news or [])
        items = []
        for n in raw[:limit]:
            c = n.get("content", n)  # yfinance changed its schema; handle both
            url = (c.get("canonicalUrl") or {}).get("url") or c.get("link")
            title = c.get("title")
            if title:
                items.append({
                    "title": title, "publisher": (c.get("provider") or {}).get("displayName") or c.get("publisher"),
                    "published": c.get("pubDate") or c.get("providerPublishTime"), "url": url,
                })
        return {"symbol": ticker, "count": len(items), "news": items}
    except Exception as e:
        return {"symbol": ticker, "count": 0, "news": [], "note": f"News unavailable: {str(e)}"}


@app.get("/cache/clear", dependencies=[Depends(verify_key)])
def clear_cache():
    with _cache_lock:
        n = len(_cache)
        _cache.clear()
    return {"cleared_entries": n}
