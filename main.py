import os
import json
import time
import math
from typing import Any, Iterable, List, Dict
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
import gspread
from google.oauth2.service_account import Credentials
from loguru import logger
from gspread.exceptions import APIError

SUMMARY_SHEET = "Kraken-Screener"
CHART_SHEET = "Kraken-Screener-chartData"

# 14 days of 15m bars = 14 * 24 * 4 = 1344
SPARKLINE_DAYS = 14
TF = "15m"
SPARKLINE_BARS = SPARKLINE_DAYS * 24 * 4

# Indicator windows
RSI_LEN = 14
SMA_240 = 240
SMA_720 = 720
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Kraken pagination safety
FETCH_LIMIT = 500

# Sheets write pacing
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "15"))  # symbols per batch write
BACKOFF_BASE = 30  # seconds, exponential backoff starting point


# ----------------------------
# Utilities / sanitizers
# ----------------------------
def sanitize_value(v: Any) -> Any:
    """Convert NaN/Inf to '', numpy scalars to python scalars.
    Keep strings (including formulas) as-is so Sheets can parse them."""
    if isinstance(v, str):
        return v
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        return v if math.isfinite(v) else ""
    return v


def sanitize_row(row: Iterable[Any]) -> list:
    return [sanitize_value(x) for x in row]


def column_index_to_letter(idx: int) -> str:
    # 1-based index to Excel/Sheets column letter
    letters = ""
    while idx > 0:
        idx, remainder = divmod(idx - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def last_float(series: pd.Series) -> float:
    """Return last numeric value as float or np.nan if missing/non-finite."""
    if series is None or len(series) == 0:
        return np.nan
    val = series.iloc[-1]
    if pd.isna(val):
        return np.nan
    try:
        f = float(val)
        return f if math.isfinite(f) else np.nan
    except Exception:
        return np.nan


def sheets_update_with_backoff(ws, *, values, range_name, value_input_option="RAW", max_retries=5):
    """Wrapper around worksheet.update with exponential backoff on 429."""
    delay = BACKOFF_BASE
    for attempt in range(max_retries):
        try:
            return ws.update(values=values, range_name=range_name, value_input_option=value_input_option)
        except APIError as e:
            msg = str(e).lower()
            if "quota exceeded" in msg or "429" in msg or "rate limit" in msg:
                logger.warning(f"Sheets quota hit; backing off {delay}s (attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("Exceeded max retries on Sheets update due to quota throttling.")


# ----------------------------
# Sheets setup
# ----------------------------
def load_sheets_client():
    creds_json = os.getenv("GOOGLE_CREDS_JSON")
    if not creds_json:
        raise RuntimeError("GOOGLE_CREDS_JSON env var is required.")
    info = json.loads(creds_json)
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(credentials)
    return gc


def open_or_create_sheet(gc, name):
    try:
        sh = gc.open(name)
    except gspread.SpreadsheetNotFound:
        sh = gc.create(name)
    return sh


def get_or_create_worksheet(sh, title, rows=2000, cols=50):
    try:
        ws = sh.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=str(rows), cols=str(cols))
    return ws


def ensure_headers(ws):
    # Only write A1:M1; do NOT clear, so columns N+ remain untouched
    headers = [
        "Symbol",           # A
        "Last Price",       # B
        "% Down from ATH",  # C
        "PL% 1D",           # D
        "PL% 7D",           # E
        "24h Volume",       # F
        "RSI14",            # G
        "SMA240",           # H
        "SMA720",           # I
        "MACD",             # J
        "MACD Signal",      # K
        "MACD Hist",        # L
        "Sparkline",        # M
    ]
    sheets_update_with_backoff(ws, values=[headers], range_name="A1:M1", value_input_option="USER_ENTERED")


# ----------------------------
# Exchange helpers
# ----------------------------
def make_exchange():
    api_key = os.getenv("KRAKEN_API_KEY")
    secret = os.getenv("KRAKEN_API_SECRET")
    exchange = ccxt.kraken(
        {
            "enableRateLimit": True,
            "rateLimit": 1000,
            "apiKey": api_key,
            "secret": secret,
            "options": {"adjustForTimeDifference": True},
        }
    )
    exchange.load_markets()
    return exchange


def usd_markets(exchange: ccxt.Exchange) -> List[str]:
    syms = []
    for mkt in exchange.markets.values():
        if not mkt.get("active", True):
            continue
        if mkt.get("spot", True) and mkt.get("quote") == "USD":
            syms.append(mkt["symbol"])  # e.g., 'BTC/USD'
    return sorted(set(syms))


def fetch_ohlcv_all(
    exchange: ccxt.Exchange, symbol: str, timeframe: str, since_ms: int, limit: int = FETCH_LIMIT
) -> list:
    all_rows = []
    since = since_ms
    while True:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except ccxt.NetworkError as e:
            logger.warning(f"Network error on {symbol} {timeframe}: {e}; retrying in 5s")
            time.sleep(5)
            continue
        except ccxt.BaseError as e:
            logger.error(f"Exchange error on {symbol} {timeframe}: {e}")
            break
        if not batch:
            break
        all_rows.extend(batch)
        # Next page
        new_since = batch[-1][0] + 1
        if new_since <= since:
            break
        since = new_since
        time.sleep(max(0.001, exchange.rateLimit / 1000))
        if len(all_rows) > 100000:  # hard cap
            break
    return all_rows


def to_df(ohlcv: list) -> pd.DataFrame:
    if not ohlcv:
        return (
            pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            .set_index(pd.Index([], name="timestamp"))
        )
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


# ----------------------------
# Indicators & metrics
# ----------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["RSI14"] = ta.rsi(df["close"], length=RSI_LEN)
    df["SMA240"] = ta.sma(df["close"], length=SMA_240)
    df["SMA720"] = ta.sma(df["close"], length=SMA_720)

    macd = ta.macd(df["close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    if macd is not None and not macd.empty:
        df["MACD"] = macd.iloc[:, 0]
        df["MACD_signal"] = macd.iloc[:, 1]
        df["MACD_hist"] = macd.iloc[:, 2]
    else:
        df["MACD"] = np.nan
        df["MACD_signal"] = np.nan
        df["MACD_hist"] = np.nan

    return df


def compute_pl(df: pd.DataFrame, days: int) -> float:
    if df.empty:
        return np.nan
    end = df["close"].dropna()
    if end.empty:
        return np.nan
    end_price = end.iloc[-1]
    delta_minutes = days * 24 * 60
    bars_back = int(delta_minutes / 15)
    if len(end) <= bars_back:
        return np.nan
    start_price = end.iloc[-1 - bars_back]
    if start_price == 0 or pd.isna(start_price):
        return np.nan
    return (float(end_price) - float(start_price)) / float(start_price) * 100.0


def fetch_ath(exchange: ccxt.Exchange, symbol: str) -> float:
    try:
        since_dt = datetime(2014, 1, 1, tzinfo=timezone.utc)
        since_ms = int(since_dt.timestamp() * 1000)
        rows = fetch_ohlcv_all(exchange, symbol, timeframe="1d", since_ms=since_ms, limit=FETCH_LIMIT)
        dfd = to_df(rows)
        if dfd.empty:
            return np.nan
        return float(dfd["high"].max())
    except Exception as e:
        logger.warning(f"ATH fetch failed for {symbol}: {e}")
        return np.nan


def fetch_24h_volume(exchange: ccxt.Exchange, symbol: str, df15: pd.DataFrame) -> float:
    try:
        t = exchange.fetch_ticker(symbol)
        vol = t.get("baseVolume")
        if vol is not None:
            return float(vol)
    except Exception:
        pass
    try:
        recent = df15.iloc[-96:]
        return float(recent["volume"].sum()) if not recent.empty else np.nan
    except Exception:
        return np.nan


# ----------------------------
# Sheet writers (batched)
# ----------------------------
def write_chart_headers(ws_chart, symbols: List[str]):
    """Single call to write the header row (symbols) for the chart sheet."""
    if not symbols:
        return
    col_end = column_index_to_letter(len(symbols))
    values = [symbols]  # one row
    sheets_update_with_backoff(
        ws_chart,
        values=values,
        range_name=f"A1:{col_end}1",
        value_input_option="USER_ENTERED",
    )


def write_chart_block(ws_chart, start_col_idx: int, closes_by_symbol: Dict[str, List[float]], symbol_order: List[str]):
    """Write a contiguous block of chart data for a batch of symbols in one request.
    closes_by_symbol[sym] -> list of floats (length SPARKLINE_BARS), no NaNs.
    """
    if not symbol_order:
        return
    end_col_idx = start_col_idx + len(symbol_order) - 1
    start_letter = column_index_to_letter(start_col_idx)
    end_letter = column_index_to_letter(end_col_idx)

    # Build rows (row-major) so gspread can update a rectangle range:
    # rows: SPARKLINE_BARS, cols: len(symbol_order)
    rows = []
    for r in range(SPARKLINE_BARS):
        row = []
        for sym in symbol_order:
            arr = closes_by_symbol.get(sym, [])
            val = float(arr[r]) if r < len(arr) and math.isfinite(arr[r]) else ""
            row.append(val)
        rows.append(row)

    sheets_update_with_backoff(
        ws_chart,
        values=rows,
        range_name=f"{start_letter}2:{end_letter}{SPARKLINE_BARS+1}",
        value_input_option="RAW",
    )


def write_summary_block(ws_sum, start_row_idx: int, start_col_idx: int, metrics_by_symbol: Dict[str, Dict[str, Any]], symbol_order: List[str]):
    """Write a contiguous block of summary rows for a batch of symbols in one request.
    Only updates A..M (does not touch N+)."""
    if not symbol_order:
        return
    rows = []
    for i, sym in enumerate(symbol_order):
        col_idx = start_col_idx + i
        col_letter = column_index_to_letter(col_idx)
        spark_formula = f"=SPARKLINE('{CHART_SHEET}'!{col_letter}2:{col_letter}{SPARKLINE_BARS+1})"

        m = metrics_by_symbol[sym]

        # Clean display: drop '/USD' in column A
        base_sym = sym.split('/')[0]

        row = [
            base_sym,                  # A
            m.get("last_price"),       # B
            m.get("pct_down_from_ath"),# C
            m.get("pl1d"),             # D
            m.get("pl7d"),             # E
            m.get("vol24h"),           # F
            m.get("rsi14"),            # G
            m.get("sma240"),           # H
            m.get("sma720"),           # I
            m.get("macd"),             # J
            m.get("macd_signal"),      # K
            m.get("macd_hist"),        # L
            spark_formula,             # M
        ]
        rows.append(sanitize_row(row))

    end_row_idx = start_row_idx + len(symbol_order) - 1
    sheets_update_with_backoff(
        ws_sum,
        values=rows,
        range_name=f"A{start_row_idx}:M{end_row_idx}",
        value_input_option="USER_ENTERED",
    )


# ----------------------------
# Main process (single cycle)
# ----------------------------
def process_once(exchange: ccxt.Exchange, sh):
    ws_sum = get_or_create_worksheet(sh, SUMMARY_SHEET, rows=6000, cols=100)
    ws_chart = get_or_create_worksheet(sh, CHART_SHEET, rows=SPARKLINE_BARS + 10, cols=3000)
    ensure_headers(ws_sum)  # only writes A1:M1

    symbols = usd_markets(exchange)
    logger.info(f"Found {len(symbols)} Kraken USD spot markets")

    # Write chart header row once per cycle (all symbols at once)
    write_chart_headers(ws_chart, symbols)

    # Bars needed for indicators + sparkline
    margin_bars = max(SMA_720, MACD_SLOW + MACD_SIGNAL + 10, RSI_LEN + 10)
    need_bars = SPARKLINE_BARS + margin_bars
    since_dt = datetime.now(timezone.utc) - timedelta(minutes=15 * need_bars)
    since_ms = int(since_dt.timestamp() * 1000)

    for chunk_start in range(0, len(symbols), BATCH_SIZE):
        chunk_syms = symbols[chunk_start:chunk_start + BATCH_SIZE]
        start_col_idx = 1 + chunk_start
        start_row_idx = 2 + chunk_start

        closes_by_symbol: Dict[str, List[float]] = {}
        metrics_by_symbol: Dict[str, Dict[str, Any]] = {}

        for sym in chunk_syms:
            logger.info(f"Processing {sym}")
            ohlcv15 = fetch_ohlcv_all(exchange, sym, TF, since_ms, limit=FETCH_LIMIT)
            df15 = to_df(ohlcv15)
            if df15.empty:
                logger.warning(f"No 15m data for {sym}")
                closes_by_symbol[sym] = []
                metrics_by_symbol[sym] = {
                    "last_price": "",
                    "pct_down_from_ath": "",
                    "pl1d": "",
                    "pl7d": "",
                    "vol24h": "",
                    "rsi14": "",
                    "sma240": "",
                    "sma720": "",
                    "macd": "",
                    "macd_signal": "",
                    "macd_hist": "",
                }
                continue

            df15 = compute_indicators(df15)
            closes = df15["close"].dropna().iloc[-SPARKLINE_BARS:]
            closes_by_symbol[sym] = [float(x) for x in closes.tolist()]

            last_price = last_float(df15["close"])
            pl1d = compute_pl(df15, 1)
            pl7d = compute_pl(df15, 7)
            vol24h = fetch_24h_volume(exchange, sym, df15)

            ath = fetch_ath(exchange, sym)
            pct_down_ath = ((last_price - ath) / ath * 100.0) if (ath and ath > 0 and math.isfinite(last_price)) else np.nan

            rsi14 = last_float(df15["RSI14"])
            sma240 = last_float(df15["SMA240"])
            sma720 = last_float(df15["SMA720"])
            macd = last_float(df15["MACD"])
            macd_signal = last_float(df15["MACD_signal"])
            macd_hist = last_float(df15["MACD_hist"])

            metrics_by_symbol[sym] = {
                "last_price": last_price,
                "pct_down_from_ath": pct_down_ath,
                "pl1d": pl1d,
                "pl7d": pl7d,
                "vol24h": vol24h,
                "rsi14": rsi14,
                "sma240": sma240,
                "sma720": sma720,
                "macd": macd,
                "macd_signal": macd_signal,
                "macd_hist": macd_hist,
            }

            # Gentle pacing for Kraken
            time.sleep(0.05)

        # Write chart block for this batch
        write_chart_block(ws_chart, start_col_idx, closes_by_symbol, chunk_syms)

        # Write summary rows (A..M) for this batch
        write_summary_block(ws_sum, start_row_idx, start_col_idx, metrics_by_symbol, chunk_syms)

        # Small pause between batches to avoid per-minute caps
        time.sleep(1.0)

    logger.info("Update cycle complete.")


def main():
    sheets_name = os.getenv("GOOGLE_SHEETS_NAME", "Trading Log")

    logger.add(lambda msg: print(msg, end=""))

    gc = load_sheets_client()
    sh = open_or_create_sheet(gc, sheets_name)

    exchange = make_exchange()

    started = datetime.now(timezone.utc)
    logger.info(f"\n===== Single cycle start {started.isoformat()} =====")
    try:
        process_once(exchange, sh)
    except Exception as e:
        logger.exception(f"Cycle error: {e}")
        # Optionally re-raise if you want cron to see a non-zero exit code:
        # raise
    ended = datetime.now(timezone.utc)
    logger.info(f"Single cycle complete at {ended.isoformat()}")


if __name__ == "__main__":
    main()
