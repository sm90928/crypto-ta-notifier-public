# Crypto TA Notifier (GitHub Actions)

Daily notifier for BTC/ETH/BNB/SOL vs USDT using Binance public data.
Calculates RSI(14), MACD(12/26/9), Bollinger(20,2), SMA20/50/200, ATR(14).  
Outputs one of **ENTER / ADD / REDUCE / EXIT / HOLD** and sends to Telegram.

## Run on GitHub Actions
The workflow runs **09:00** and **22:35 Asia/Taipei** daily.

1. Create a **Private** repo and upload these files.
2. Go to **Settings → Secrets and variables → Actions** and add secrets:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
3. Check **Actions** tab; you can **Run workflow** once to test.

## Config (env)
`SYMBOLS` (default: `BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT`), `INTERVAL` (`1d`), `LIMIT` (`300`), and thresholds like `VOL_SPIKE_MULT`, `SMA_FAST/MID/SLOW`, etc. can be overridden in the workflow `env`.

## 20/80 Ladder plan (tips embedded in messages)
- **ENTER**: buy 20% now; place 80% ladder at **−8% / −15% / −25% / −35%**
- **ADD**: use only the next tranche on dips; for **breakout + retest**, repurpose the deepest tranche for a **10–15%** confirmation add
- **REDUCE**: 10–30% near weakness at resistance; trail with **20D MA** or **3×ATR**
- **EXIT**: 20–40% on daily trend breaks; additional 30–50% on weekly invalidation  
On `EXIT`, message includes concrete trailing stop examples: **20D SMA** and **close − 3×ATR(14)**.
