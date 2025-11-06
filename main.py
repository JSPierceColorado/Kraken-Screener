import os
import json
import time
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta

import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
import gspread
from google.oauth2.service_account import Credentials
from loguru import logger

SUMMARY_SHEET = "Kraken-Screener"
CHART_SHEET = "Kraken-Screener-chartData"

SPARKLINE_DAYS = 14
TF = "15m"
SPARKLINE_BARS = SPARKLINE_DAYS * 24 * 4

RSI_LEN = 14
SMA_240 = 240
SMA_720 = 720
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
FETCH_LIMIT = 500


def load_sheets_client():
    creds_json = os.getenv("GOOGLE_CREDS_JSON")
    if not creds_json:
        raise RuntimeError("GOOGLE_CREDS_JSON env var is required.")
    info = json.loads(creds_json)
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
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


def column_index_to_letter(idx: int) -> str:
    letters = ''
    while idx > 0:
        idx, remainder = divmod(idx - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def make_exchange():
    api_key = os.getenv("KRAKEN_API_KEY")
    secret = os.getenv("KRAKEN_API_SECRET")
    exchange = ccxt.kraken({
        "enableRateLimit": True,
        "rateLimit": 1000,
        "apiKey": api_key,
        "secret": secret,
        "options": {"adjustForTimeDifference": True},
    })
    exchange.load_markets()
    return exchange


def usd_markets(exchange: ccxt.Exchange) -> list[str]:
    syms = []
    for mkt in exchange.markets.values():
        if not mkt.get("active", True):
            continue
        if mkt.get("spot", True) and mkt.get("quote") == "USD":
            syms.append(mkt["symbol"])
    return sorted(set(syms))


def fetch_ohlcv_all(exchange: ccxt.Exchange, symbol: str, timeframe: str, since_ms: int, limit: int = FETCH_LIMIT) -> list:
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
        new_since = batch[-1][0] + 1
        if new_since <= since:
            break
        since = new_since
        time.sleep(exchange.rateLimit / 1000)
        if len(all_rows) > 100000:
            break
    return all_rows


def to_df(ohlcv: list) -> pd.DataFrame:
    if not ohlcv:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]).set_index(pd.Index([], name="timestamp"))
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


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
    if start_price == 0 or np.isnan(start_price):
        return np.nan
    return (end_price - start_price) / start_price * 100.0


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


def ensure_headers(ws):
    headers = [
        "Symbol", "Last Price", "% Down from ATH", "PL% 1D", "PL% 7D", "PL% 14D",
        "24h Volume", "RSI14", "SMA240", "SMA720", "MACD", "MACD Signal", "MACD Hist", "Sparkline"
    ]
    existing = ws.row_values(1)
    if existing != headers:
        ws.clear()
        ws.update([headers])


def write_chart_column(ws_chart, col_idx: int, symbol: str, closes: pd.Series):
    col_letter = column_index_to_letter(col_idx)
    ws_chart.update_cell(1, col_idx, symbol)
    if closes is None or closes.empty:
        return
    values = [[float(x)] for x in closes.tolist()]
    ws_chart.update(f"{col_letter}2:{col_letter}{len(values)+1}", values)


def write_summary_row(ws_sum, row_idx: int, symbol: str, metrics: dict, spark_col_idx: int, nrows_chart: int):
    col_letter = column_index_to_letter(spark_col_idx)
    spark_formula = f"=SPARKLINE('{CHART_SHEET}'!{col_letter}2:{col_letter}{nrows_chart})"

    row = [
        symbol,
        metrics.get("last_price"),
        metrics.get("pct_down_from_ath"),
        metrics.get("pl1d"),
        metrics.get("pl7d"),
        metrics.get("pl14d"),
        metrics.get("vol24h"),
        metrics.get("rsi14"),
        metrics.get("sma240"),
        metrics.get("sma720"),
        metrics.get("macd"),
        metrics.get("macd_signal"),
        metrics.get("macd_hist"),
        spark_formula,
    ]
    ws_sum.update(f"A{row_idx}:N{row_idx}", [row])


def process_once(exchange: ccxt.Exchange, sh):
    ws_sum = get_or_create_worksheet(sh, SUMMARY_SHEET, rows=5000, cols=100)
    ws_chart = get_or_create_worksheet(sh, CHART_SHEET, rows=2000, cols=2000)
    ensure_headers(ws_sum)

    symbols = usd_markets(exchange)
    logger.info(f"Found {len(symbols)} Kraken USD spot markets")

    margin_bars = max(SMA_720, MACD_SLOW + MACD_SIGNAL + 10, RSI_LEN + 10)
    need_bars = SPARKLINE_BARS + margin_bars
    since_dt = datetime.now(timezone.utc) - timedelta(minutes=15 * need_bars)
    since_ms = int(since_dt.timestamp() * 1000)

    col_idx = 1
    nrows_chart = SPARKLINE_BARS

    for i, sym in enumerate(symbols, start=2):
        logger.info(f"Processing {sym}")
        ohlcv15 = fetch_ohlcv_all(exchange, sym, TF, since_ms, limit=FETCH_LIMIT)
        df15 = to_df(ohlcv15)
        if df15.empty:
            logger.warning(f"No 15m data for {sym}")
            continue

        df15 = compute_indicators(df15)
        closes = df15["close"].dropna().iloc[-SPARKLINE_BARS:]

        last_price = float(df15["close"].iloc[-1])
        pl1d = compute_pl(df15, 1)
        pl7d = compute_pl(df15, 7)
        pl14d = compute_pl(df15, 14)
        vol24h = fetch_24h_volume(exchange, sym, df15)

        ath = fetch_ath(exchange, sym)
        pct_down_ath = float((last_price - ath) / ath * 100.0) if ath and ath > 0 else np.nan

        rsi14 = float(df15["RSI14"].iloc[-1]) if not np.isnan(df15["RSI14"].iloc[-1]) else np.nan
        sma240 = float(df15["SMA240"].iloc[-1]) if not np.isnan(df15["SMA240"].iloc[-1]) else np.nan
        sma720 = float(df15["SMA720"].iloc[-1]) if not np.isnan(df15["SMA720"].iloc[-1]) else np.nan
        macd = float(df15["MACD"].iloc[-1]) if not np.isnan(df15["MACD"].iloc[-1]) else np.nan
        macd_signal = float(df15["MACD_signal"].iloc[-1]) if not np.isnan(df15["MACD_signal"].iloc[-1]) else np.nan
        macd_hist = float(df15["MACD_hist"].iloc[-1]) if not np.isnan(df15["MACD_hist"].iloc[-1]) else np.nan

        write_chart_column(ws_chart, col_idx, sym, closes)

        metrics = {
            "last_price": last_price,
            "pct_down_from_ath": pct_down_ath,
            "pl1d": pl1d,
            "pl7d": pl7d,
            "pl14d": pl14d,
            "vol24h": vol24h,
            "rsi14": rsi14,
            "sma240": sma240,
            "sma720": sma720,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
        }

        write_summary_row(ws_sum, i, sym, metrics, col_idx, nrows_chart + 1)

        col_idx += 1
        time.sleep(0.3)

    logger.info("Update cycle complete.")


def main():
    sheets_name = os.getenv("GOOGLE_SHEETS_NAME", "Trading Log")
    interval = int(os.getenv("UPDATE_INTERVAL_SECONDS", "900"))

    logger.add(lambda msg: print(msg, end=""))

    gc = load_sheets_client()
    sh = open_or_create_sheet(gc, sheets_name)

    exchange = make_exchange()

    while True:
        started = datetime.now(timezone.utc)
        logger.info(f"\\n===== Cycle start {started.isoformat()} =====")
        try:
            process_once(exchange, sh)
        except Exception as e:
            logger.exception(f"Cycle error: {e}")
        ended = datetime.now(timezone.utc)
        elapsed = (ended - started).total_seconds()

        now = datetime.now(timezone.utc)
        minutes = now.minute
        sleep_to_next_quarter = ((15 - (minutes % 15)) % 15) * 60 - now.second
        sleep_seconds = max(interval, sleep_to_next_quarter)
        if sleep_seconds < 60:
            sleep_seconds = interval
        logger.info(f"Sleeping {int(sleep_seconds)}s...")
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
