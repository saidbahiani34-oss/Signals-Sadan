import os
import time
import math
import json
import logging
from datetime import datetime, timezone

import requests
import pandas as pd

# --- Basic logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

BINANCE_BASE = "https://api.binance.com"

# --- Config (ENV) ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # e.g. @YourChannelUsername

# Cadence
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "900"))  # new signals scan
MONITOR_INTERVAL_SEC = int(os.getenv("MONITOR_INTERVAL_SEC", "120"))  # signal tracking

# Universe
TOP_N = int(os.getenv("TOP_N", "150"))
MIN_QUOTE_VOL_USDT = float(os.getenv("MIN_QUOTE_VOL_USDT", "10000000"))
MIN_LISTING_DAYS = int(os.getenv("MIN_LISTING_DAYS", "30"))

# Anti-spam
COOLDOWN_BARS_15M = int(os.getenv("COOLDOWN_BARS_15M", "8"))
OPEN_SIGNALS_MAX = int(os.getenv("OPEN_SIGNALS_MAX", "80"))
SIGNAL_EXPIRY_MIN = int(os.getenv("SIGNAL_EXPIRY_MIN", "720"))  # pending expiry

# Modes
ENABLE_CONSERVATIVE = os.getenv("ENABLE_CONSERVATIVE", "1") == "1"
ENABLE_AGGRESSIVE = os.getenv("ENABLE_AGGRESSIVE", "1") == "1"

# ATR% thresholds
ATR_LOW = float(os.getenv("ATR_LOW", "1.0"))
ATR_HIGH = float(os.getenv("ATR_HIGH", "2.5"))

# TPs (ATR multiples)
CONS_TP1 = float(os.getenv("CONS_TP1", "1.2"))
CONS_TP2 = float(os.getenv("CONS_TP2", "2.4"))
AGGR_TP1 = float(os.getenv("AGGR_TP1", "0.9"))
AGGR_TP2 = float(os.getenv("AGGR_TP2", "1.8"))

# SL (ATR multiples)
CONS_SL = float(os.getenv("CONS_SL", "1.5"))
AGGR_SL = float(os.getenv("AGGR_SL", "1.3"))

# Trailing (Chandelier k)
CONS_K = float(os.getenv("CONS_K", "3.0"))
AGGR_K = float(os.getenv("AGGR_K", "2.2"))

# Optional: send trailing updates (can be noisy)
SEND_TRAIL_UPDATES = os.getenv("SEND_TRAIL_UPDATES", "0") == "1"
TRAIL_UPDATE_STEP_ATR = float(os.getenv("TRAIL_UPDATE_STEP_ATR", "0.3"))

# Persist state
STATE_FILE = os.getenv("STATE_FILE", "state.json")

# Exclusions (avoid some leveraged tokens)
EXCLUDE_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")


def http_get(path, params=None, timeout=15):
    url = BINANCE_BASE + path
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def http_post(url, payload=None, timeout=20):
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_top_symbols_usdt_spot(top_n=150, min_quote_vol=10_000_000):
    tickers = http_get("/api/v3/ticker/24hr")
    out = []
    for t in tickers:
        sym = t.get("symbol")
        if not sym or not sym.endswith("USDT"):
            continue
        if sym.endswith(EXCLUDE_SUFFIXES):
            continue
        qv = float(t.get("quoteVolume", 0.0))
        if qv < min_quote_vol:
            continue
        out.append((sym, qv))

    out.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in out[:top_n]]


def fetch_all_prices():
    data = http_get("/api/v3/ticker/price")
    return {d["symbol"]: float(d["price"]) for d in data if "symbol" in d and "price" in d}


def fetch_klines(symbol, interval, limit=210, start_time_ms=None):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time_ms is not None:
        params["startTime"] = int(start_time_ms)
    data = http_get("/api/v3/klines", params=params)
    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open","high","low","close","volume","quote_asset_volume","taker_buy_base","taker_buy_quote"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


def listing_age_days(symbol):
    df = fetch_klines(symbol, "1d", limit=1, start_time_ms=0)
    first = df.loc[0, "open_time"].to_pydatetime()
    now = datetime.now(timezone.utc)
    return (now - first).days


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, math.nan)
    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def atr_bucket(atr_pct):
    if atr_pct < ATR_LOW:
        return "Low"
    if atr_pct <= ATR_HIGH:
        return "Med"
    return "High"


def btc_gate_allows_aggressive():
    try:
        df = fetch_klines("BTCUSDT", "1h", limit=260)
        close = df["close"]
        ema200 = ema(close, 200)
        ema50 = ema(close, 50)
        a = atr(df, 14)
        atr_pct = float(a.iloc[-1] / close.iloc[-1] * 100)

        cond = (close.iloc[-1] < ema200.iloc[-1]) and (ema50.iloc[-1] < ema200.iloc[-1]) and (atr_pct > 1.2)
        return not cond
    except Exception as e:
        logging.warning("BTC gate check failed: %s", e)
        return True


def conservative_signal(symbol):
    df15 = fetch_klines(symbol, "15m", limit=220)
    df1h = fetch_klines(symbol, "1h", limit=260)
    df4h = fetch_klines(symbol, "4h", limit=260)

    c15 = df15["close"]
    c1h = df1h["close"]
    c4h = df4h["close"]

    ema200_1h = ema(c1h, 200)
    ema200_4h = ema(c4h, 200)

    if c1h.iloc[-1] <= ema200_1h.iloc[-1]:
        return None

    a15 = atr(df15, 14)
    atr_val = float(a15.iloc[-1])
    atr_pct = float(atr_val / c15.iloc[-1] * 100)
    if atr_pct > ATR_HIGH:
        return None

    lookback = 30
    res = float(df15["high"].iloc[-(lookback+2):-2].max())
    prev_close = float(c15.iloc[-2])
    last_low = float(df15["low"].iloc[-1])
    last_close = float(c15.iloc[-1])

    vol = df15["volume"]
    vol_sma20 = float(vol.rolling(20).mean().iloc[-1])
    vol_ratio = float(vol.iloc[-1] / vol_sma20) if vol_sma20 and vol_sma20 > 0 else 1.0

    breakout = prev_close > res
    retest = (last_low <= res * 1.001) and (last_close > res)
    vol_ok = vol_ratio >= 1.2

    ema20 = ema(c15, 20)
    ema50 = ema(c15, 50)
    r = rsi(c15, 14)
    pullback = (ema20.iloc[-1] > ema50.iloc[-1]) and (df15["low"].iloc[-1] <= float(ema20.iloc[-1]) * 1.001)
    confirm = (last_close > float(ema20.iloc[-1])) and (float(r.iloc[-1]) > 50)

    trigger = None
    if breakout and retest and vol_ok:
        trigger = "Breakout+Retest"
    elif pullback and confirm and vol_ratio >= 1.1:
        trigger = "EMA Pullback"

    if not trigger:
        return None

    entry = last_close
    entry_a = entry - 0.15 * atr_val
    entry_b = entry + 0.05 * atr_val
    sl = entry - CONS_SL * atr_val
    tp1 = entry + CONS_TP1 * atr_val
    tp2 = entry + CONS_TP2 * atr_val

    conf = 6
    if vol_ratio >= 1.5:
        conf += 1
    if c4h.iloc[-1] > ema200_4h.iloc[-1]:
        conf += 1

    return {
        "mode": "Conservative",
        "symbol": symbol,
        "trigger": trigger,
        "entry_a": entry_a,
        "entry_b": entry_b,
        "entry_ref": entry,
        "sl_init": sl,
        "tp1": tp1,
        "tp2": tp2,
        "atr": atr_val,
        "atr_pct": atr_pct,
        "atr_bucket": atr_bucket(atr_pct),
        "confidence": min(conf, 10),
        "trail_k": CONS_K,
    }


def aggressive_signal(symbol):
    df15 = fetch_klines(symbol, "15m", limit=220)
    df1h = fetch_klines(symbol, "1h", limit=260)

    c15 = df15["close"]
    c1h = df1h["close"]

    ema200_1h = ema(c1h, 200)
    if float(c1h.iloc[-1]) < float(ema200_1h.iloc[-1]) and (c1h.iloc[-3:] < ema200_1h.iloc[-3:]).all():
        return None

    a15 = atr(df15, 14)
    atr_val = float(a15.iloc[-1])
    atr_pct = float(atr_val / c15.iloc[-1] * 100)

    vol = df15["volume"]
    vol_sma20 = float(vol.rolling(20).mean().iloc[-1])
    vol_ratio = float(vol.iloc[-1] / vol_sma20) if vol_sma20 and vol_sma20 > 0 else 1.0

    r = rsi(c15, 14)

    lookback = 20
    hh = float(df15["high"].iloc[-(lookback+1):-1].max())
    last_close = float(c15.iloc[-1])

    momentum_break = (last_close > hh) and (vol_ratio >= 1.0) and (float(r.iloc[-1]) > 55)

    df1h_48 = df1h.iloc[-48:]
    level = float((df1h_48["high"].max() + df1h_48["low"].min()) / 2)
    reclaim = (float(c15.iloc[-2]) < level) and (last_close > level) and (vol_ratio >= 1.0)

    trigger = None
    if momentum_break:
        trigger = "Momentum Break"
    elif reclaim:
        trigger = "Range Reclaim"

    if not trigger:
        return None

    entry = last_close
    entry_a = entry - 0.20 * atr_val
    entry_b = entry + 0.07 * atr_val
    sl = entry - AGGR_SL * atr_val
    tp1 = entry + AGGR_TP1 * atr_val
    tp2 = entry + AGGR_TP2 * atr_val

    conf = 5
    if vol_ratio >= 1.5:
        conf += 1
    if atr_pct <= ATR_HIGH:
        conf += 1

    return {
        "mode": "Aggressive",
        "symbol": symbol,
        "trigger": trigger,
        "entry_a": entry_a,
        "entry_b": entry_b,
        "entry_ref": entry,
        "sl_init": sl,
        "tp1": tp1,
        "tp2": tp2,
        "atr": atr_val,
        "atr_pct": atr_pct,
        "atr_bucket": atr_bucket(atr_pct),
        "confidence": min(conf, 10),
        "trail_k": AGGR_K,
    }


def fmt(x):
    if x >= 100:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:,.4f}"
    return f"{x:,.8f}"


def pair_pretty(symbol):
    return symbol.replace("USDT", "/USDT")


def build_signal_message(sig):
    pair = pair_pretty(sig["symbol"])
    return (
        f"SPOT BUY | Binance USDT | Mode: {sig['mode']}\n"
        f"Pair: {pair}\n"
        f"TF: 15m trigger | 1h filter | 4h context\n"
        f"Entry Zone: {fmt(sig['entry_a'])} — {fmt(sig['entry_b'])}\n"
        f"SL: {fmt(sig['sl_init'])} (ATR-based)\n"
        f"TP1: {fmt(sig['tp1'])}  |  TP2: {fmt(sig['tp2'])}\n"
        f"TP3: Trailing (Chandelier k={sig['trail_k']})\n"
        f"ATR%: {sig['atr_pct']:.2f}% ({sig['atr_bucket']})\n"
        f"Confidence: {sig['confidence']}/10\n"
        f"Reason: {sig['trigger']}\n"
        f"Status: PENDING\n"
        f"\n"
        f"ملاحظة: بعد TP1: انقل SL إلى Entry - 0.1×ATR. بعد TP2: فعّل Trailing."
    )


def build_update_message(s, update_type, price):
    # Style C: عربي + مصطلحات/أرقام إنجليزية
    pair = pair_pretty(s["symbol"])

    if update_type == "ACTIVE":
        return (
            f"UPDATE | {pair} | ✅ ACTIVE\n"
            f"Mode: {s['mode']} | Price: {fmt(price)}\n"
            f"Entry: {fmt(s['entry_ref'])} | SL: {fmt(s['sl_current'])}\n"
            f"TP1: {fmt(s['tp1'])} | TP2: {fmt(s['tp2'])}"
        )

    if update_type == "TP1":
        return (
            f"UPDATE | {pair} | 🎯 TP1 HIT\n"
            f"Price: {fmt(price)}\n"
            f"SL moved → {fmt(s['sl_current'])} (Entry - 0.1×ATR)\n"
            f"Next: TP2 {fmt(s['tp2'])}"
        )

    if update_type == "TRAIL_ON":
        return (
            f"UPDATE | {pair} | 🧲 TRAILING ON\n"
            f"Price: {fmt(price)}\n"
            f"Chandelier k={s['trail_k']} | TSL: {fmt(s['tsl'])}\n"
            f"HH: {fmt(s['hh'])}"
        )

    if update_type == "TRAIL_UP":
        return (
            f"UPDATE | {pair} | ↗️ TRAILING UPDATED\n"
            f"Price: {fmt(price)}\n"
            f"TSL: {fmt(s['tsl'])} | HH: {fmt(s['hh'])}"
        )

    if update_type == "EXIT_SL":
        return (
            f"UPDATE | {pair} | 🛑 EXIT (SL)\n"
            f"Price: {fmt(price)}\n"
            f"SL: {fmt(s['sl_current'])}"
        )

    if update_type == "EXIT_TSL":
        return (
            f"UPDATE | {pair} | 🛑 EXIT (Trailing Stop)\n"
            f"Price: {fmt(price)}\n"
            f"TSL: {fmt(s['tsl'])} | HH: {fmt(s['hh'])}"
        )

    if update_type == "EXPIRED":
        return (
            f"UPDATE | {pair} | ⏳ EXPIRED\n"
            f"لم يتم تفعيل الدخول خلال {SIGNAL_EXPIRY_MIN} دقيقة."
        )

    return None


def tg_send(text, reply_to_message_id=None):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    if reply_to_message_id is not None:
        payload["reply_to_message_id"] = int(reply_to_message_id)

    return http_post(url, payload=payload, timeout=25)


def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"cooldown": {}, "signals": {}}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def now_ts():
    return datetime.now(timezone.utc).timestamp()


def is_in_entry_zone(price, a, b):
    lo = min(a, b)
    hi = max(a, b)
    return lo <= price <= hi


def monitor_signals(state, prices):
    signals = state.get("signals", {})
    if not signals:
        return

    to_delete = []
    for sid, s in signals.items():
        sym = s["symbol"]
        price = prices.get(sym)
        if price is None:
            continue

        status = s.get("status", "PENDING")
        created_ts = float(s.get("created_ts", 0))
        age_min = (now_ts() - created_ts) / 60 if created_ts else 0

        # Pending expiry
        if status == "PENDING" and age_min >= SIGNAL_EXPIRY_MIN:
            msg = build_update_message(s, "EXPIRED", price)
            if msg:
                tg_send(msg, reply_to_message_id=s.get("msg_id"))
            to_delete.append(sid)
            continue

        # Activation
        if status == "PENDING":
            if is_in_entry_zone(price, s["entry_a"], s["entry_b"]):
                s["status"] = "ACTIVE"
                s["activated_ts"] = now_ts()
                s["hh"] = price
                s["sl_current"] = s["sl_init"]
                s["tp1_hit"] = False
                s["tp2_hit"] = False
                s["trail_on"] = False
                s["tsl"] = None
                s["last_trail_announce"] = None

                msg = build_update_message(s, "ACTIVE", price)
                if msg:
                    tg_send(msg, reply_to_message_id=s.get("msg_id"))
            continue

        # Active/trailing management
        if status in ("ACTIVE", "TP1", "TRAILING"):
            # Update HH
            s["hh"] = max(float(s.get("hh", price)), price)

            # SL check (only if trailing not on)
            sl_current = float(s.get("sl_current", s["sl_init"]))
            if price <= sl_current and not s.get("trail_on", False):
                msg = build_update_message(s, "EXIT_SL", price)
                if msg:
                    tg_send(msg, reply_to_message_id=s.get("msg_id"))
                to_delete.append(sid)
                continue

            # TP1
            if (not s.get("tp1_hit", False)) and price >= float(s["tp1"]):
                s["tp1_hit"] = True
                s["status"] = "TP1"
                # Move SL to Entry - 0.1*ATR
                s["sl_current"] = float(s["entry_ref"]) - 0.1 * float(s["atr"])

                msg = build_update_message(s, "TP1", price)
                if msg:
                    tg_send(msg, reply_to_message_id=s.get("msg_id"))

            # TP2 => trailing on
            if (not s.get("tp2_hit", False)) and price >= float(s["tp2"]):
                s["tp2_hit"] = True
                s["status"] = "TRAILING"
                s["trail_on"] = True
                # Init TSL
                s["tsl"] = float(s["hh"]) - float(s["trail_k"]) * float(s["atr"])

                msg = build_update_message(s, "TRAIL_ON", price)
                if msg:
                    tg_send(msg, reply_to_message_id=s.get("msg_id"))

            # Trailing update + exit
            if s.get("trail_on", False):
                old_tsl = float(s.get("tsl", 0.0))
                new_tsl = float(s["hh"]) - float(s["trail_k"]) * float(s["atr"])
                if new_tsl > old_tsl:
                    s["tsl"] = new_tsl

                    if SEND_TRAIL_UPDATES:
                        step = TRAIL_UPDATE_STEP_ATR * float(s["atr"])
                        last_ann = s.get("last_trail_announce")
                        if last_ann is None or (new_tsl - float(last_ann)) >= step:
                            s["last_trail_announce"] = new_tsl
                            msg = build_update_message(s, "TRAIL_UP", price)
                            if msg:
                                tg_send(msg, reply_to_message_id=s.get("msg_id"))

                if price <= float(s["tsl"]):
                    msg = build_update_message(s, "EXIT_TSL", price)
                    if msg:
                        tg_send(msg, reply_to_message_id=s.get("msg_id"))
                    to_delete.append(sid)
                    continue

        signals[sid] = s

    for sid in to_delete:
        signals.pop(sid, None)

    state["signals"] = signals


def main_loop():
    state = load_state()
    cooldown = state.get("cooldown", {})
    signals = state.get("signals", {})

    last_scan = 0.0

    while True:
        try:
            prices = fetch_all_prices()
            monitor_signals(state, prices)

            # Scan for new signals periodically
            tnow = now_ts()
            if (tnow - last_scan) >= SCAN_INTERVAL_SEC:
                allow_aggr = btc_gate_allows_aggressive()
                symbols = fetch_top_symbols_usdt_spot(TOP_N, MIN_QUOTE_VOL_USDT)

                # If too many open signals, pause new ones
                if len(state.get("signals", {})) >= OPEN_SIGNALS_MAX:
                    logging.info("Open signals >= %d, skip new scan cycle", OPEN_SIGNALS_MAX)
                    last_scan = tnow
                    save_state(state)
                    time.sleep(MONITOR_INTERVAL_SEC)
                    continue

                now_dt = datetime.now(timezone.utc)
                logging.info("New-signal scan | symbols=%d | aggressive_allowed=%s | open=%d", len(symbols), allow_aggr, len(state.get("signals", {})))

                sent = 0
                for sym in symbols:
                    # listing age filter
                    try:
                        age = listing_age_days(sym)
                        if age < MIN_LISTING_DAYS:
                            continue
                    except Exception:
                        continue

                    # cooldown check
                    last_ts = cooldown.get(sym)
                    if last_ts:
                        since = now_dt.timestamp() - float(last_ts)
                        if since < COOLDOWN_BARS_15M * 15 * 60:
                            continue

                    sig = None
                    if ENABLE_CONSERVATIVE:
                        sig = conservative_signal(sym)

                    if (sig is None) and ENABLE_AGGRESSIVE and allow_aggr:
                        sig = aggressive_signal(sym)

                    if sig is None:
                        continue

                    # send initial signal
                    msg = build_signal_message(sig)
                    resp = tg_send(msg)
                    msg_id = None
                    try:
                        msg_id = resp.get("result", {}).get("message_id")
                    except Exception:
                        msg_id = None

                    # register tracking state
                    sid = f"{sym}-{int(now_dt.timestamp())}"
                    sig_state = {
                        **sig,
                        "id": sid,
                        "created_ts": now_dt.timestamp(),
                        "status": "PENDING",
                        "msg_id": msg_id,
                        "sl_current": sig["sl_init"],
                    }

                    signals = state.get("signals", {})
                    signals[sid] = sig_state
                    state["signals"] = signals

                    cooldown[sym] = now_dt.timestamp()
                    sent += 1

                    # soft pacing
                    time.sleep(0.12)

                    if len(state.get("signals", {})) >= OPEN_SIGNALS_MAX:
                        break

                state["cooldown"] = cooldown
                save_state(state)
                logging.info("New-signal scan done | sent=%d | open=%d", sent, len(state.get("signals", {})))

                last_scan = tnow

        except Exception as e:
            logging.exception("Loop error: %s", e)

        # monitor pace
        time.sleep(MONITOR_INTERVAL_SEC)


if __name__ == "__main__":
    main_loop()
