#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Daily TA Notifier (GitHub Actions friendly)
- Binance API 旋轉: api → api1 → api2 → api3
- 後備: CoinGecko OHLC (daily, 全歷史); 無量能時自動關閉放量條件
- 隱藏錯誤訊息中的 URL
- 放寬最少K棒門檻：Binance >=220；CoinGecko >=60
- 指標：RSI(14), MACD(12/26/9), Bollinger(20,2), SMA20/50/200, ATR(14)
- 訊號：ENTER / ADD / REDUCE / EXIT / HOLD
- 文案：20/80 提示 + EXIT 顯示 20D / 3×ATR
"""
import os, sys, math, time, json, requests
import datetime as dt
from typing import List, Dict, Any
try:
    import pandas as pd
    import numpy as np
except Exception:
    print("Requires: pandas numpy requests", file=sys.stderr)
    raise

BINANCE_ENV_BASE = os.environ.get("BINANCE_API_BASE", "https://api.binance.com")
SYMBOLS = os.environ.get("SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT").split(",")
INTERVAL = os.environ.get("INTERVAL", "1d")
LIMIT = int(os.environ.get("LIMIT", "300"))

VOL_SPIKE_MULT = float(os.environ.get("VOL_SPIKE_MULT", "1.20"))
BOLL_STD = float(os.environ.get("BOLL_STD", "2.0"))
RSI_PERIOD = int(os.environ.get("RSI_PERIOD", "14"))
ATR_PERIOD = int(os.environ.get("ATR_PERIOD", "14"))
SMA_FAST = int(os.environ.get("SMA_FAST", "20"))
SMA_MID = int(os.environ.get("SMA_MID", "50"))
SMA_SLOW = int(os.environ.get("SMA_SLOW", "200"))

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
OUTPUT_JSON = os.environ.get("OUTPUT_JSON", "ta_notifier_last.json")

def _dedupe(seq):
    seen=set(); out=[]
    for x in seq:
        if x and x not in seen:
            out.append(x); seen.add(x)
    return out

def send_telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(msg); return
    url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload={"chat_id":TELEGRAM_CHAT_ID,"text":msg,"parse_mode":"Markdown"}
    try:
        r=requests.post(url,json=payload,timeout=12)
        if not r.ok: print(f"[Telegram] {r.status_code} {r.text}", file=sys.stderr)
    except Exception as e:
        print(f"[Telegram] Exception: {e}", file=sys.stderr)

def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def sma(s,n): return s.rolling(window=n,min_periods=n).mean()
def rsi(s,n=14):
    d=s.diff(); up=(d.where(d>0,0.0)).abs(); dn=(-d.where(d<0,0.0)).abs()
    avg_up=up.ewm(alpha=1/n,adjust=False).mean(); avg_dn=dn.ewm(alpha=1/n,adjust=False).mean()
    rs=avg_up/(avg_dn.replace(0,np.nan)); out=100-(100/(1+rs)); return out.fillna(50.0)
def macd(s,fast=12,slow=26,signal=9):
    m=ema(s,fast)-ema(s,slow); sig=ema(m,signal); hist=m-sig; return m,sig,hist
def bollinger(s,n=20,k=2.0):
    mid=sma(s,n); std=s.rolling(window=n,min_periods=n).std()
    up=mid+k*std; lo=mid-k*std; bw=(up-lo)/mid; return up,mid,lo,bw
def atr(h,l,c,n=14):
    pc=c.shift(1); tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()
def vol_avg(v,n=20): return v.rolling(window=n,min_periods=n).mean()

def get_binance_bases():
    return _dedupe([BINANCE_ENV_BASE,"https://api.binance.com","https://api1.binance.com","https://api2.binance.com","https://api3.binance.com"])

def fetch_klines_binance(symbol, interval, limit=300):
    cols=["open_time","open","high","low","close","volume","close_time","qav","trades","taker_base","taker_quote","ignore"]
    last_err=None
    for base in get_binance_bases():
        try:
            url=f"{base}/api/v3/klines"
            r=requests.get(url,params={"symbol":symbol,"interval":interval,"limit":limit},timeout=20)
            r.raise_for_status(); data=r.json()
            df=pd.DataFrame(data,columns=cols)
            df["open_time"]=pd.to_datetime(df["open_time"],unit="ms")
            df["close_time"]=pd.to_datetime(df["close_time"],unit="ms")
            for c in ["open","high","low","close","volume"]:
                df[c]=pd.to_numeric(df[c],errors="coerce")
            df.set_index("close_time",inplace=True)
            df["__source__"]="binance"
            return df
        except Exception as e:
            last_err=e; continue
    raise last_err or RuntimeError("Unknown Binance error")

COINGECKO_ID={"BTC":"bitcoin","BTCUSDT":"bitcoin","ETH":"ethereum","ETHUSDT":"ethereum",
              "BNB":"binancecoin","BNBUSDT":"binancecoin","SOL":"solana","SOLUSDT":"solana"}

def fetch_ohlc_coingecko(symbol, limit=300):
    base=symbol.upper().replace("USDT","")
    cg_id=COINGECKO_ID.get(symbol.upper()) or COINGECKO_ID.get(base)
    if not cg_id: raise ValueError(f"No CoinGecko id mapping for {symbol}")
    # 抓「全歷史」確保足夠K棒
    url=f"https://api.coingecko.com/api/v3/coins/{cg_id}/ohlc"
    r=requests.get(url,params={"vs_currency":"usd","days":"max"},timeout=25)
    r.raise_for_status(); data=r.json()
    if not isinstance(data,list) or not data: raise ValueError("Empty OHLC from CoinGecko")
    df=pd.DataFrame(data,columns=["t","open","high","low","close"])
    df["close_time"]=pd.to_datetime(df["t"],unit="ms"); df.drop(columns=["t"],inplace=True)
    df["volume"]=np.nan
    for c in ["open","high","low","close"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df.set_index("close_time",inplace=True)
    if len(df)>limit: df=df.tail(limit)
    df["__source__"]="coingecko"
    return df

def fetch_market_df(symbol, interval, limit=300):
    try:
        return fetch_klines_binance(symbol, interval, limit)
    except Exception as e:
        if interval=="1d":
            try:
                return fetch_ohlc_coingecko(symbol, limit)
            except Exception as e2:
                raise RuntimeError(f"Binance+CoinGecko failed: {e} | {e2}")
        raise

def plan_tip_for_action(action: str) -> str:
    a=(action or "").upper()
    if a=="ENTER": return "20/80 建倉：先買 20%，其餘 80% 分四階掛 −8%/−15%/−25%/−35%。"
    if a=="ADD":   return "加碼：只動用下一階掛單；若突破回測站穩，可把最深一階改為 10–15% 的『突破加碼』。"
    if a=="REDUCE":return "減碼：壓力區轉弱或貼上軌回中軌下，先了結 10–30%，用 20D 或 3×ATR 追蹤。"
    if a=="EXIT":  return "風控：日線趨勢失效（如 <SMA200 或 MACD 轉空且 <SMA50）減 20–40%；週線失效再減 30–50%。"
    return "觀望：維持 20/80 階梯，不追單；等回測或突破回測站穩再動。"

def evaluate(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    c,h,l,v=df["close"],df["high"],df["low"],df["volume"]
    rsi14=rsi(c,RSI_PERIOD); macd_line,macd_sig,macd_hist=macd(c,12,26,9)
    bb_u,bb_m,bb_l,bb_bw=bollinger(c,SMA_FAST,BOLL_STD)
    sma20,sma50,sma200=sma(c,SMA_FAST),sma(c,SMA_MID),sma(c,SMA_SLOW)
    atr14,vol20=atr(h,l,c,ATR_PERIOD),vol_avg(v,20)
    df=df.copy()
    df["rsi14"]=rsi14; df["macd"]=macd_line; df["macd_signal"]=macd_sig; df["macd_hist"]=macd_hist
    df["bb_u"]=bb_u; df["bb_m"]=bb_m; df["bb_l"]=bb_l; df["bb_bw"]=bb_bw
    df["sma20"]=sma20; df["sma50"]=sma50; df["sma200"]=sma200
    df["atr14"]=atr14; df["vol20"]=vol20
    row=df.iloc[-1]; prev=df.iloc[-2] if len(df)>=2 else row
    price=row["close"]; above_bb_u=price>row["bb_u"]; rsi_hot=row["rsi14"]>=70; rsi_strong=row["rsi14"]>=55
    macd_bull=(row["macd_hist"]>0) and (row["macd"]>row["macd_signal"])
    macd_bear=(row["macd_hist"]<0) and (row["macd"]<row["macd_signal"])
    vol_spike=False if (isinstance(row["vol20"],float) and math.isnan(row["vol20"])) else (row["volume"]>(row["vol20"]*VOL_SPIKE_MULT))
    action,reasons="HOLD",[]
    if (price<row["sma200"]) or (macd_bear and price<row["sma50"]):
        action="EXIT"; reasons.append("Trend break: close < SMA200 or MACD bear & close < SMA50")
    elif (prev["close"]>prev["bb_m"] and price<row["bb_m"]) or (rsi_hot and (row["macd_hist"]<prev["macd_hist"])):
        action="REDUCE"; reasons.append("Momentum cooling: fell below BB mid or RSI>70 with weakening MACD")
    elif (prev["close"]<prev["bb_m"] and price>row["bb_m"] and rsi_strong and row["macd_hist"]>prev["macd_hist"]) \
         or (above_bb_u and macd_bull and vol_spike and row["bb_bw"]>prev["bb_bw"]):
        action="ADD"; reasons.append("Reclaim BB mid with strong RSI & rising MACD, or high-volume breakout")
    elif (price>row["sma50"] and macd_bull and row["rsi14"]>=50):
        action="ENTER"; reasons.append("Uptrend intact: price > SMA50, MACD bull, RSI>=50")
    ref={
        "price":round(float(price),2),
        "sma20":round(float(row["sma20"]),2) if not math.isnan(row["sma20"]) else None,
        "sma50":round(float(row["sma50"]),2) if not math.isnan(row["sma50"]) else None,
        "sma200":round(float(row["sma200"]),2) if not math.isnan(row["sma200"]) else None,
        "bb_upper":round(float(row["bb_u"]),2) if not math.isnan(row["bb_u"]) else None,
        "bb_mid":round(float(row["bb_m"]),2) if not math.isnan(row["bb_m"]) else None,
        "bb_lower":round(float(row["bb_l"]),2) if not math.isnan(row["bb_l"]) else None,
        "rsi14":round(float(row["rsi14"]),2),
        "macd":round(float(row["macd"]),4),
        "macd_signal":round(float(row["macd_signal"]),4),
        "macd_hist":round(float(row["macd_hist"]),4),
        "atr14":round(float(row["atr14"]),2) if not math.isnan(row["atr14"]) else None,
        "vol":None if (isinstance(row["volume"],float) and math.isnan(row["volume"])) else round(float(row["volume"]),2),
        "vol20":None if (isinstance(row["vol20"],float) and math.isnan(row["vol20"])) else round(float(row["vol20"]),2),
        "ref_stop20d":round(float(row["sma20"]),2) if not math.isnan(row["sma20"]) else None,
        "ref_stop_atr3":(round(float(price-3*row["atr14"]),2) if (not math.isnan(row["atr14"])) else None),
        "source": df["__source__"].iloc[-1] if "__source__" in df.columns else "unknown",
    }
    return {"symbol":symbol,"action":action,"reasons":reasons,"ref":ref}

def plan_tip_for_action_line(action: str) -> str:
    tip=plan_tip_for_action(action); return f"  策略提示：{tip}" if tip else ""

def format_message(results: List[Dict[str, Any]], interval: str) -> str:
    now=dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines=[f"*Crypto TA Notifier* — {now}", f"_Interval: {interval}_", ""]
    priority={"EXIT":0,"REDUCE":1,"ADD":2,"ENTER":3,"HOLD":4,"ERROR":5}
    results_sorted=sorted(results,key=lambda r:priority.get(r.get("action",""),9))
    for r in results_sorted:
        ref=r.get("ref",{}); action=r.get("action","HOLD")
        lines.append(f"*{r.get('symbol','?')}* → *{action}*")
        reasons=r.get("reasons") or []
        if reasons:
            cleaned=[]
            for x in reasons:
                x=str(x)
                if " for url:" in x: x=x.split(" for url:")[0].strip()
                cleaned.append(x)
            lines.append("• " + "; ".join(cleaned))
        if ref and action!="ERROR":
            lines.append(f"  price {ref.get('price')} | RSI {ref.get('rsi14')} | MACD {ref.get('macd')} / {ref.get('macd_signal')} (hist {ref.get('macd_hist')})")
            lines.append(f"  BB mid {ref.get('bb_mid')} | upper {ref.get('bb_upper')} | SMA50 {ref.get('sma50')} | SMA200 {ref.get('sma200')}")
            if ref.get("vol20") is not None:
                lines.append(f"  Vol {ref.get('vol')} vs 20d {ref.get('vol20')}")
            if ref.get("source")=="coingecko":
                lines.append("  source: coingecko (no volume)")
        tip_line=plan_tip_for_action_line(action)
        if tip_line: lines.append(tip_line)
        if action=="EXIT":
            lines.append(f"  移動停損參考：20D {ref.get('ref_stop20d')} ／ 3×ATR {ref.get('ref_stop_atr3')}")
        lines.append("")
    return "\n".join(lines).strip()

def main():
    results=[]
    for sym in SYMBOLS:
        try:
            df=fetch_market_df(sym, INTERVAL, LIMIT)
            # 放寬門檻：Binance 要 >=220；CoinGecko >=60（足夠算大多數指標）
            src=df["__source__"].iloc[-1] if "__source__" in df.columns else "unknown"
            min_bars = 220 if src=="binance" else 60
            if df is None or len(df) < min_bars:
                raise ValueError("Not enough data returned")
            res=evaluate(df, sym)
            results.append(res); time.sleep(0.2)
        except Exception as e:
            err=str(e)
            if " for url:" in err: err=err.split(" for url:")[0].strip()
            results.append({"symbol":sym,"action":"ERROR","reasons":[err],"ref":{}})
    msg=format_message(results, INTERVAL); send_telegram(msg)
    try:
        with open(OUTPUT_JSON,"w",encoding="utf-8") as f:
            json.dump({"ts":dt.datetime.utcnow().isoformat(),"interval":INTERVAL,"results":results},f,ensure_ascii=False,indent=2)
    except Exception as e:
        print(f"[WARN] could not write JSON: {e}", file=sys.stderr)

if __name__=="__main__":
    main()
