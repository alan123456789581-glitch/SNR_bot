"""
backtest_sensitivity.py

參數掃描 + 樣本內／樣本外驗證。

為什麼要分樣本內／樣本外：參數是在「看過結果的那份資料」上挑的，挑出來的組合在那份資料上
一定好看（哪怕只是運氣）。正確的做法是：
  1. 只用「樣本內」（前 70% 的時間）的結果挑參數
  2. 挑完之後，只看一次「樣本外」（後 30%）；樣本外沒有撐住就是過擬合，不要回頭再調參數，
     再調就等於把樣本外也拿來挑參數了
t 值 = 平均每筆淨R ÷ (標準差 ÷ √筆數)：大約 2 以上才算跟 0 有明顯差異；樣本小的時候特別要看。

只抓一次資料，重複跑多組參數。每組都是完整回測，掃很多組會跑很久。
"""

import logging

import numpy as np
import pandas as pd

import snr_strategy
from backtest_snr import (
    fetch_full_history, run_backtest, build_data_exchange, add_fee_adjusted_r,
    SYMBOL, TIMEFRAME_H4, TIMEFRAME_M5, BACKTEST_DAYS,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

# 每一項是一組參數。lookback = 「明顯高低點」要比前面幾根K線都更高／更低；
# min_stop = 停損下限（None = 用 snr_strategy.MIN_STOP_DISTANCE_PCT）。
PARAM_GRID = [
    # 假設二的 A/B 對照（experiments.md）：同一份資料、同一個程式，只差 4H 方向有沒有 24 小時有效期限
    {"lookback": 20, "min_stop": None, "prox": None, "expiry": float("inf")},   # 基準：方向一直有效
    {"lookback": 20, "min_stop": None, "prox": None, "expiry": None},           # 假設二：用 snr_strategy.H4_BIAS_EXPIRY_HOURS（24 小時）
]
# prox=None → 用 snr_strategy.H4_ZONE_PROXIMITY_PCT（目前關閉）；expiry=float("inf") → 明確關閉有效期限
IN_SAMPLE_FRACTION = 0.7


def _stats(df):
    """df 已經有 r_multiple / r_multiple_after_fee。"""
    if df.empty:
        return {"n": 0, "win": np.nan, "gross": np.nan, "net": np.nan, "t": np.nan, "cum": 0.0, "dd": 0.0}
    net = df["r_multiple_after_fee"]
    cum = net.cumsum()
    sd = net.std(ddof=1) if len(net) > 1 else np.nan
    return {
        "n": len(df),
        "win": (df["result"] == "win").mean(),
        "gross": df["r_multiple"].mean(),
        "net": net.mean(),
        "t": net.mean() / (sd / np.sqrt(len(net))) if sd and sd > 0 else np.nan,
        "cum": cum.iloc[-1],
        "dd": (cum - cum.cummax()).min(),
    }


def summarize(trades, split_time):
    if not trades:
        empty = _stats(pd.DataFrame())
        return {"all": empty, "is": empty, "oos": empty}
    df = add_fee_adjusted_r(pd.DataFrame(trades))
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    return {
        "all": _stats(df),
        "is": _stats(df[df["entry_time"] <= split_time]),
        "oos": _stats(df[df["entry_time"] > split_time]),
    }


def _fmt(s):
    if s["n"] == 0:
        return "無交易"
    return (f"{s['n']:5d}筆 勝率{s['win']:5.1%} 毛均R{s['gross']:+.3f} 淨均R{s['net']:+.3f} "
            f"(t={s['t']:+.1f}) 累積{s['cum']:+8.1f} 回落{s['dd']:7.1f}")


if __name__ == "__main__":
    exchange = build_data_exchange()
    until_ms = exchange.milliseconds()
    since_ms = until_ms - BACKTEST_DAYS * 24 * 60 * 60 * 1000

    print(f"抓取 {BACKTEST_DAYS} 天的 4H / 5m 資料（只抓一次，重複用於每組參數）...")
    df_4h = fetch_full_history(exchange, SYMBOL, TIMEFRAME_H4, since_ms, until_ms)
    df_5m = fetch_full_history(exchange, SYMBOL, TIMEFRAME_M5, since_ms, until_ms)
    print(f"4H 資料筆數: {len(df_4h)} | 5m 資料筆數: {len(df_5m)}")

    start, end = df_5m.index[0], df_5m.index[-1]
    split_time = start + (end - start) * IN_SAMPLE_FRACTION
    print(f"樣本內: {start} → {split_time} | 樣本外: {split_time} → {end}")
    print(f"區間畫法 ZONE_MODE={snr_strategy.ZONE_MODE}，淨R已扣手續費、未計滑價\n")

    rows = []
    for p in PARAM_GRID:
        trades = run_backtest(df_4h, df_5m, lookback=p["lookback"], min_stop_distance_pct=p["min_stop"],
                              h4_proximity_pct=p.get("prox"),
                              h4_bias_expiry_hours=p.get("expiry"))
        st = summarize(trades, split_time)
        min_stop = p["min_stop"] if p["min_stop"] is not None else snr_strategy.MIN_STOP_DISTANCE_PCT
        prox = p.get("prox") if p.get("prox") is not None else snr_strategy.H4_ZONE_PROXIMITY_PCT
        prox_label = "無限制" if prox is None or prox == float("inf") else f"{prox:.2%}"
        ex = p.get("expiry") if p.get("expiry") is not None else snr_strategy.H4_BIAS_EXPIRY_HOURS
        ex_label = "無限制" if ex is None or ex == float("inf") else f"{ex}小時"
        print(f"lookback={p['lookback']:<3} 停損下限={min_stop:.2%} 4H位置限制={prox_label} 方向有效期限={ex_label}")
        print(f"   全期    {_fmt(st['all'])}")
        print(f"   樣本內  {_fmt(st['is'])}")
        print(f"   樣本外  {_fmt(st['oos'])}\n")
        for name, s in st.items():
            rows.append({"lookback": p["lookback"], "min_stop_pct": min_stop, "4H位置限制": prox_label, "方向有效期限": ex_label, "範圍": name, **s})

    pd.DataFrame(rows).to_csv("snr_param_sensitivity.csv", index=False)
    print("完整結果已存成 snr_param_sensitivity.csv")