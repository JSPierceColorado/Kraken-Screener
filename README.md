# Kraken USD Spot Screener → Google Sheets

A Python 3 tool that scans **Kraken** USD spot markets via **ccxt**, computes trading metrics (RSI, SMA240/SMA720, MACD, P/L since 1d/7d, ATH drawdown, 24h volume), and writes a clean, filterable dashboard to **Google Sheets** — complete with per‑symbol **sparklines** built from 15‑minute closes.

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-stable-brightgreen" />
  <img alt="python" src="https://img.shields.io/badge/python-3.11%2B-blue" />
  <img alt="exchange" src="https://img.shields.io/badge/exchange-Kraken-6f42c1" />
  <img alt="license" src="https://img.shields.io/badge/license-MIT-lightgrey" />
</p>

---

## ✨ What it does

* **Discovers markets:** enumerates active **Kraken** spot pairs with `USD` quote.
* **Fetches candles:** paginates **15m OHLCV** for the last 14 days (default) plus margin for indicators.
* **Computes metrics:** RSI(14), SMA240, SMA720, MACD(12/26/9), P/L since 1d/7d, % down from ATH, 24h base volume.
* **Writes two sheets:**

  * `Kraken-Screener` — summary dashboard with metrics + sparkline per symbol
  * `Kraken-Screener-chartData` — raw close prices for sparklines (one column per symbol)
* **Batched writes + backoff:** efficient updates with exponential backoff on Google Sheets quota (429) and gentle pacing on Kraken.
* **Clock‑aligned loop:** wakes on 15‑minute boundaries for tidy, fresh data.

---

## 🧭 Sheet layout

### `Kraken-Screener` (summary)

Columns A–M are managed by the script (N+ are yours to use):

| Col | Field               | Notes                                                           |
| --- | ------------------- | --------------------------------------------------------------- |
| A   | **Symbol**          | Base asset only (e.g., `BTC`)                                   |
| B   | **Last Price**      | last 15m close                                                  |
| C   | **% Down from ATH** | relative to historical daily high                               |
| D   | **PL% 1D**          | % change vs price 1 day ago                                     |
| E   | **PL% 7D**          | % change vs price 7 days ago                                    |
| F   | **24h Volume**      | base volume (ticker if available, else sum of last 96×15m bars) |
| G   | **RSI14**           | RSI on 15m closes                                               |
| H   | **SMA240**          | 240‑bar SMA (≈2.5 trading days on 15m)                          |
| I   | **SMA720**          | 720‑bar SMA                                                     |
| J   | **MACD**            | MACD line                                                       |
| K   | **MACD Signal**     | signal line                                                     |
| L   | **MACD Hist**       | histogram                                                       |
| M   | **Sparkline**       | from chartData sheet                                            |

### `Kraken-Screener-chartData` (sparklines)

* Row **1**: symbols (full pair names like `BTC/USD`)
* Rows **2..N**: one column per symbol with **15m closes** (last 14 days → 1,344 rows by default)

Sparkline formula used in the summary sheet (column **M**):

```gs
=SPARKLINE('Kraken-Screener-chartData'!<COL>2:<COL><LASTROW>)
```

> The script fills the correct column letter automatically for each symbol.

---

## 📦 Requirements

* Python **3.11+**
* A **Google Service Account** with access to the spreadsheet
* **ccxt**, **pandas**, **numpy**, **pandas-ta**, **gspread**, **loguru**, **python-dateutil**

Install:

```bash
pip install -r requirements.txt
```

*Minimal packages if you roll your own:*

```
ccxt
pandas
numpy
pandas-ta
gspread
google-auth
loguru
python-dateutil
```

---

## 🔧 Configuration (env vars)

| Variable                  | Default       | Required | Description                                                |
| ------------------------- | ------------- | :------: | ---------------------------------------------------------- |
| `GOOGLE_CREDS_JSON`       | —             |     ✅    | JSON **string** for the Google service account credentials |
| `GOOGLE_SHEETS_NAME`      | `Trading Log` |          | Spreadsheet name to create/open                            |
| `KRAKEN_API_KEY`          | —             |          | Optional; public data works without keys                   |
| `KRAKEN_API_SECRET`       | —             |          | Optional; not needed for public ticker/ohlcv               |
| `BATCH_SIZE`              | `15`          |          | Symbols per batch write to Sheets                          |
| `UPDATE_INTERVAL_SECONDS` | `900`         |          | Minimum sleep between cycles (seconds)                     |

Indicator & sparkline tunables (change source if desired):

* `SPARKLINE_DAYS` (default **14**) → 15m bars = `days * 24 * 4`
* `RSI_LEN` (14), `SMA_240` (240), `SMA_720` (720), `MACD_FAST` (12), `MACD_SLOW` (26), `MACD_SIGNAL` (9)

### Example `.env`

```env
GOOGLE_CREDS_JSON='{"type":"service_account",...}'
GOOGLE_SHEETS_NAME=Trading Log
BATCH_SIZE=20
UPDATE_INTERVAL_SECONDS=900
```

> Tip: Minify your service account JSON before pasting into `GOOGLE_CREDS_JSON`:
>
> ```bash
> python -c "import json,sys; print(json.dumps(json.load(sys.stdin)))" < service-account.json
> ```

---

## ▶️ Run it

### Local

```bash
export $(grep -v '^#' .env | xargs)  # optional
python main.py
```

### Docker (example)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "main.py"]
```

```bash
docker build -t kraken-screener .
docker run --rm \
  -e GOOGLE_CREDS_JSON \
  -e GOOGLE_SHEETS_NAME \
  -e BATCH_SIZE \
  -e UPDATE_INTERVAL_SECONDS \
  kraken-screener
```

### GitHub Actions (scheduled)

Create `.github/workflows/kraken_screener.yml`:

```yaml
name: Kraken Screener

on:
  workflow_dispatch: {}
  schedule:
    # Every 15 min on weekdays (UTC). Adjust to your timezone if needed.
    - cron: "*/15 * * * 1-5"

jobs:
  run:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run screener
        env:
          GOOGLE_CREDS_JSON: ${{ secrets.GOOGLE_CREDS_JSON }}
          GOOGLE_SHEETS_NAME: ${{ secrets.GOOGLE_SHEETS_NAME }}
          BATCH_SIZE: ${{ secrets.BATCH_SIZE }}
          UPDATE_INTERVAL_SECONDS: ${{ secrets.UPDATE_INTERVAL_SECONDS }}
          KRAKEN_API_KEY: ${{ secrets.KRAKEN_API_KEY }}
          KRAKEN_API_SECRET: ${{ secrets.KRAKEN_API_SECRET }}
        run: |
          python main.py
```

> GitHub cron is **UTC**. If you’re in America/Denver, adjust schedule times accordingly.

Badge (after first run):

```md
![build](https://github.com/<you>/<repo>/actions/workflows/kraken_screener.yml/badge.svg)
```

---

## 🧮 Metrics & logic

* **RSI14** on 15m closes
* **SMA240 / SMA720** on 15m closes
* **MACD(12,26,9)** on 15m closes (line, signal, histogram)
* **P/L since 1d/7d** computed from closes `96` and `672` bars back
* **ATH drawdown** based on daily highs since 2014‑01‑01
* **24h Volume** from exchange ticker when available; otherwise sum of last 96×15m volumes

**Data hygiene**

* Non‑finite values are sanitized to empty strings (Sheets‑friendly)
* Numpy scalars converted to Python scalars

**Throughput & stability**

* Kraken paginated OHLCV with rate‑limit awareness
* Google Sheets writes are **batched** and protected by exponential backoff (429/quotas)
* Gentle pacing between symbols and batches

---

## 🧱 Architecture (at a glance)

```
┌───────── Kraken (ccxt) ──────────┐
│  list USD spot markets           │
│  fetch 15m OHLCV + daily highs   │
└───────────────┬──────────────────┘
                │ pandas + pandas_ta
         compute indicators & metrics
                │
┌───────────────▼──────────────────┐
│ Google Sheets (gspread)          │
│  - Kraken-Screener               │
│  - Kraken-Screener-chartData     │
└──────────────────────────────────┘
```

---

## 🔒 Security

* Treat `GOOGLE_CREDS_JSON` and any Kraken API keys as **secrets** (GitHub Secrets, Docker/K8s secret stores).
* Never commit credentials. Use environment variables or a secrets manager.

---

## 🧰 Troubleshooting

* **`GOOGLE_CREDS_JSON env var is required.`** – Provide a minified JSON string; ensure the service account has **Editor** access to the sheet.
* **Empty data / few rows** – Kraken may not have long history for new pairs; increase lookback where relevant.
* **429 / quota exceeded** – The script already backs off; consider lowering `BATCH_SIZE` or reducing schedule frequency.
* **Misaligned times** – GitHub Actions uses **UTC**; the runtime loop aligns to 15m boundaries automatically.

---

## 🤝 Contributing

PRs and issues welcome. For significant changes, please open a discussion first.

## 📜 License

MIT — see `LICENSE`.
