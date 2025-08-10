#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Daily TA Notifier (GitHub Actions friendly)
Pairs: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT (configurable)
Indicators: RSI(14), MACD(12/26/9), Bollinger(20,2), SMA(20/50/200), ATR(14)
Signals: ENTER / ADD / REDUCE / EXIT / HOLD
Outputs: Telegram message (if env set) or console
"""
import os, sys, math, time, json, requests
import datetime as dt
from typing import List, Dict, Any
try:
    import pandas as pd
    import numpy as np
except Exception as e:
    print("Requires: pandas numpy requests", file=sys.stderr)
    raise

BINANCE = os.environ.get("BINANCE_API_BASE", "https://api.binance.com")
SYMBOLS = os.environ.get("SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT").split(",")
INTERVAL = os.environ.get("INTERVAL","1d")
LIMIT = int(os.environ.get("LIMIT","300"))

VOL_SPIKE_MULT = float(os.environ.get("VOL_SPIKE_MULT","1.20"))
BOLL_STD = float(os.environ.get("BOLL_STD","2.0"))
RSI_PERIOD = int(os.environ.get("RSI_PERIOD","14"))
ATR_PERIOD = int(os.environ.get("ATR_PERIOD","14"))
SMA_FAST = int(os.environ.get("SMA_FAST","20"))
SMA_MID = int(os.environ.get("SMA_MID","50"))
SMA_SLOW = int(os.environ.get("SMA_SLOW","200"))

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN","")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID","")
OUTPUT_JSON = os.environ.get("OUTPUT_JSON","ta_notifier_last.json")

def fetch_klines(symbol: str, interval: str, limit: int=300):
    url = f"{BINANCE}/api/v3/klines"
    r = requests.get(url, params={"symbol":symbol, "interval":interval, "limit":limit}, timeout=20)
    r.raise_for_status()
    data = r.json()
    cols = ["open_time","open","high","low","close","volume","close_time",
            "qav","trades","taker_buy_base","taker_buy_quote","ignore"]
    import pandas as pd
    df = pd.DataFrame(data, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.set_index("close_time", inplace=True)
    return df

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(window=n, min_periods=n).mean()

def rsi(s, n=14):
    d = s.diff()
    up = (d.where(d>0,0.0)).abs()
    dn = (-d.where(d<0,0.0)).abs()
    avg_up = up.ewm(alpha=1/n, adjust=False).mean()
    avg_dn = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_up / (avg_dn.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def macd(s, fast=12, slow=26, signal=9):
    m = ema(s, fast) - ema(s, slow)
    sig = ema(m, signal)
    hist = m - sig
    return m, sig, hist

def bollinger(s, n=20, k=2.0):
    mid = sma(s, n)
    std = s.rolling(window=n, min_periods=n).std()
    up = mid + k*std
    lo = mid - k*std
    bw = (up - lo) / mid
    return up, mid, lo, bw

def atr(h, l, c, n=14):
    pc = c.shift(1)
    import pandas as pd
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def vol_avg(v, n=20): return v.rolling(window=n, min_periods=n).mean()

def send_telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(msg)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if not r.ok:
            print(f"[Telegram] {r.text}", file=sys.stderr)
    except Exception as e:
        print(f"[Telegram] Exception: {e}", file=sys.stderr)

# ----------------------- Signal Logic -----------------------
def plan_tip_for_action(action: str) -> str:
    a = (action or "").upper()
    if a == "ENTER":
        return "20/80 建倉：先買 20%，其餘 80% 分四階掛 −8%/−15%/−25%/−35%。"
    if a == "ADD":
        return "加碼：只動用下一階掛單（避免連續補刀）。若有效突破並回測站穩，可把最深一階改為 10–15% 的『突破加碼』。"
    if a == "REDUCE":
        return "減碼：壓力區轉弱、或貼上軌後收回中軌下，先了結 10–30%，並用 20 日均線或 3×ATR 追蹤停利。"
    if a == "EXIT":
        return "風險控制：日線跌破趨勢（如 SMA200 或 MACD 轉空且收在 SMA50 下）減 20–40%；週線失效再減 30–50%。"
    return "觀望：維持 20/80 階梯，不追單；等待回測或突破回測站穩再動。"

def evaluate(df, symbol: str):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    rsi14 = rsi(c, RSI_PERIOD)
    macd_line, macd_sig, macd_hist = macd(c, 12, 26, 9)
    bb_u, bb_m, bb_l, bb_bw = bollinger(c, SMA_FAST, BOLL_STD)
    sma20, sma50, sma200 = sma(c, SMA_FAST), sma(c, SMA_MID), sma(c, SMA_SLOW)
    atr14, vol20 = atr(h,l,c,ATR_PERIOD), vol_avg(v,20)

    df = df.copy()
    df["rsi14"]=rsi14; df["macd"]=macd_line; df["macd_signal"]=macd_sig; df["macd_hist"]=macd_hist
    df["bb_u"]=bb_u; df["bb_m"]=bb_m; df["bb_l"]=bb_l; df["bb_bw"]=bb_bw
    df["sma20"]=sma20; df["sma50"]=sma50; df["sma200"]=sma200
    df["atr14"]=atr14; df["vol20"]=vol20
    row = df.iloc[-1]; prev = df.iloc[-2] if len(df)>=2 else row

    price = row["close"]
    above_bb_u = price > row["bb_u"]
    rsi_hot = row["rsi14"] >= 70
    rsi_strong = row["rsi14"] >= 55
    macd_bull = (row["macd_hist"] > 0) and (row["macd"] > row["macd_signal"])
    macd_bear = (row["macd_hist"] < 0) and (row["macd"] < row["macd_signal"])
    vol_spike = False if math.isnan(row["vol20"]) else (row["volume"] > (row["vol20"]*VOL_SPIKE_MULT))

    action, reasons = "HOLD", []

    # EXIT
    if (price < row["sma200"]) or (macd_bear and price < row["sma50"]):
        action = "EXIT"; reasons.append("Trend break: close < SMA200 or MACD bear & close < SMA50")
    # REDUCE
    elif (prev["close"] > prev["bb_m"] and price < row["bb_m"]) or (rsi_hot and (row["macd_hist"] < prev["macd_hist"])):
        action = "REDUCE"; reasons.append("Momentum cooling: fell below BB mid or RSI>70 with weakening MACD")
    # ADD
    elif (prev["close"] < prev["bb_m"] and price > row["bb_m"] and rsi_strong and row["macd_hist"] > prev["macd_hist"]) \
         or (above_bb_u and macd_bull and vol_spike and row["bb_bw"] > prev["bb_bw"]):
        action = "ADD"; reasons.append("Reclaim BB mid with strong RSI & rising MACD, or high-volume breakout")
    # ENTER
    elif (price > row["sma50"] and macd_bull and row["rsi14"] >= 50):
        action = "ENTER"; reasons.append("Uptrend intact: price > SMA50, MACD bull, RSI>=50")

    ref = {
        "price": round(float(price),2),
        "sma20": round(float(row["sma20"]),2) if not math.isnan(row["sma20"]) else None,
        "sma50": round(float(row["sma50"]),2) if not math.isnan(row["sma50"]) else None,
        "sma200": round(float(row["sma200"]),2) if not math.isnan(row["sma200"]) else None,
        "bb_upper": round(float(row["bb_u"]),2) if not math.isnan(row["bb_u"]) else None,
        "bb_mid": round(float(row["bb_m"]),2) if not math.isnan(row["bb_m"]) else None,
        "bb_lower": round(float(row["bb_l"]),2) if not math.isnan(row["bb_l"]) else None,
        "rsi14": round(float(row["rsi14"]),2),
        "macd": round(float(row["macd"]),4),
        "macd_signal": round(float(row["macd_signal"]),4),
        "macd_hist": round(float(row["macd_hist"]),4),
        "atr14": round(float(row["atr14"]),2) if not math.isnan(row["atr14"]) else None,
        "vol": round(float(row["volume"]),2),
        "vol20": round(float(row["vol20"]),2) if not math.isnan(row["vol20"]) else None,
        # Trailing stop references
        "ref_stop20d": round(float(row["sma20"]),2) if not math.isnan(row["sma20"]) else None,
        "ref_stop_atr3": (round(float(price - 3*row["atr14"]),2) if (not math.isnan(row["atr14"])) else None),
    }
    return {"symbol":symbol, "action":action, "reasons":reasons, "ref":ref}

def plan_tip_for_action_line(action: str) -> str:
    tip = plan_tip_for_action(action)
    return f"  策略提示：{tip}" if tip else ""

def format_message(results: List[Dict[str, Any]], interval: str) -> str:
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"*Crypto TA Notifier* — {now}", f"_Interval: {interval}_", ""]
    priority = {"EXIT":0, "REDUCE":1, "ADD":2, "ENTER":3, "HOLD":4}
    results_sorted = sorted(results, key=lambda r: priority.get(r.get("action",""), 9))
    for r in results_sorted:
        ref = r.get("ref",{}); action = r.get("action","HOLD")
        lines.append(f"*{r.get('symbol','?')}* → *{action}*")
        reasons = r.get("reasons") or []
        if reasons: lines.append("• " + "; ".join(reasons))
        if ref:
            lines.append(f"  price {ref.get('price')} | RSI {ref.get('rsi14')} | MACD {ref.get('macd')} / {ref.get('macd_signal')} (hist {ref.get('macd_hist')})")
            lines.append(f"  BB mid {ref.get('bb_mid')} | upper {ref.get('bb_upper')} | SMA50 {ref.get('sma50')} | SMA200 {ref.get('sma200')}")
            if ref.get('vol20'): lines.append(f"  Vol {ref.get('vol')} vs 20d {ref.get('vol20')}")
        tip_line = plan_tip_for_action_line(action)
        if tip_line: lines.append(tip_line)
        if action == 'EXIT' and ref:
            lines.append(f"  移動停損參考：20D {ref.get('ref_stop20d')} ／ 3×ATR {ref.get('ref_stop_atr3')}")
        lines.append("")
    return "\\n".join(lines).strip()

def main():
    results = []
    for sym in SYMBOLS:
        try:
            df = fetch_klines(sym, INTERVAL, LIMIT)
            if df is None or len(df) < 220:
                raise ValueError("Not enough data returned")
            res = evaluate(df, sym)
            results.append(res)
            time.sleep(0.2)
        except Exception as e:
            results.append({"symbol":sym, "action":"ERROR", "reasons":[str(e)], "ref":{}})
    msg = format_message(results, INTERVAL)
    send_telegram(msg)
    try:
        with open(OUTPUT_JSON,"w",encoding="utf-8") as f:
            json.dump({"ts": dt.datetime.utcnow().isoformat(), "interval":INTERVAL, "results":results}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] could not write JSON: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
