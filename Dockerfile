# Kraken screener runtime
USE_WHITELIST=false
WHITELIST=
OHLC_INTERVAL=5
ORDER_BOOK_COUNT=100
MAX_CONCURRENT_REQUESTS=8
OUTPUT_PATH=/app/output/screener_latest.csv

# Continuous run mode
RUN_CONTINUOUSLY=true
RUN_INTERVAL_SECONDS=300
RUN_ON_STARTUP=true

# Google Sheets sync
GOOGLE_SHEETS_ENABLED=true
GOOGLE_SHEETS_SYNC_ON_RUN=true
GOOGLE_SHEETS_WORKSHEET_NAME=Screener

# Use either the spreadsheet ID or the full URL
GOOGLE_SHEETS_SPREADSHEET_ID=YOUR_SPREADSHEET_ID
# GOOGLE_SHEETS_SPREADSHEET_URL=https://docs.google.com/spreadsheets/d/YOUR_SPREADSHEET_ID/edit#gid=0

# Paste the full service-account JSON into one Railway variable.
# In Railway's RAW editor, JSON values can be pasted directly.
GOOGLE_SERVICE_ACCOUNT_JSON={"type":"service_account","project_id":"...","private_key_id":"...","private_key":"-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n","client_email":"...","client_id":"...","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_x509_cert_url":"...","universe_domain":"googleapis.com"}
