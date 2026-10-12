"""
compare_live_vs_backtest.py

把實盤交易紀錄（snr_live_trades.csv）跟回測交易明細（snr_backtest_trades.csv）逐筆對帳。

用法（時間一律是 UTC；台灣時間要減 8 小時）：
    py compare_live_vs_backtest.py --start "2026-10-09 02:00:00"

正確的對帳流程：
  1. 讓機器人「連續跑、中途不重啟」，記下最後一次啟動的時間（換成 UTC）
  2. 在 backtest_snr.py 把 COMPARE_START_UTC 設成同一個時間，重跑回測（會更新
     snr_backtest_trades.csv）。回測的 BACKTEST_DAYS 要涵蓋這段期間再多留 50 天以上暖身
  3. 執行這支腳本，--start 填同一個時間

為什麼一定要這樣做：策略一次只持一筆、而且上一筆結束前不找新訊號，所以只要有一筆
交易兩邊不同，後面每一筆都會連鎖地跟著不同。起點狀態不一致，逐筆比對沒有意義。

注意：snr_live_trades.csv 裡的 signal_time / exit_time 已經是 UTC（K線時間戳記跟成交
時間戳記），不用換算；換算只用在你看 log 檔裡的時間（那是電腦本地時間）。
"""

import argparse
import os

import numpy as np
import pandas as pd

BACKTEST_COLS_OLD = [
    "h4_bias", "entry_time", "entry_price", "stop_loss", "take_profit",
    "exit_time", "exit_price", "result", "r_multiple", "r_multiple_after_fee", "cumulative_r",
]
TAKER_FEE_REF = 0.0005  # 回測目前假設的單邊吃單手續費，用來跟實盤隱含費率對照


# ---------------------------------------------------------
# 讀檔
# ---------------------------------------------------------
def load_backtest(path):
    with open(path, encoding="utf-8") as f:
        first = f.readline()
    if "entry_time" in first:
        df = pd.read_csv(path)
    else:
        # 舊版沒有表頭的格式
        df = pd.read_csv(path, header=None)
        df = df.iloc[:, : len(BACKTEST_COLS_OLD)]
        df.columns = BACKTEST_COLS_OLD[: df.shape[1]]
    for c in ("entry_time", "exit_time"):
        df[c] = pd.to_datetime(df[c])
    df = df.rename(columns={"h4_bias": "side"})
    return df.sort_values("entry_time").reset_index(drop=True)


def load_live(path):
    df = pd.read_csv(path)
    for c in ("signal_time", "exit_time"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df.sort_values("exit_time").reset_index(drop=True)


# ---------------------------------------------------------
# 配對
# ---------------------------------------------------------
def match_trades(live, bt, tol_minutes):
    """同方向、訊號K線時間差在容許範圍內的配成一對；時間差小的優先。"""
    cands = []
    for i, lr in live.iterrows():
        if pd.isna(lr["signal_time"]):
            continue
        for j, br in bt.iterrows():
            if lr["side"] != br["side"]:
                continue
            d = abs((lr["signal_time"] - br["entry_time"]).total_seconds()) / 60
            if d <= tol_minutes:
                cands.append((d, i, j))
    cands.sort()
    used_l, used_b, pairs = set(), set(), []
    for d, i, j in cands:
        if i in used_l or j in used_b:
            continue
        used_l.add(i)
        used_b.add(j)
        pairs.append((i, j, d))
    return pairs, used_l, used_b


def in_position(t, starts, ends):
    return any((s <= t <= e) for s, e in zip(starts, ends) if pd.notna(s) and pd.notna(e))


# ---------------------------------------------------------
# 滑價與手續費（只需要實盤紀錄）
# ---------------------------------------------------------
def slippage_and_fee_report(live):
    need = ["planned_entry_price", "entry_price", "stop_loss", "take_profit", "exit_price"]
    d = live.dropna(subset=need).copy()
    if d.empty:
        print("實盤紀錄缺少計畫價位，無法計算滑價。")
        return

    sign = np.where(d["side"] == "long", 1, -1)
    # 進場：做多買貴、做空賣便宜 = 不利（正值代表吃虧）
    d["entry_slip"] = sign * (d["entry_price"] - d["planned_entry_price"])
    # 出場：用最接近的那個計畫價位（停損或停利）當計畫出場價
    near_sl = (d["exit_price"] - d["stop_loss"]).abs() < (d["exit_price"] - d["take_profit"]).abs()
    planned_exit = np.where(near_sl, d["stop_loss"], d["take_profit"])
    # 出場：做多賣得比計畫低、做空買得比計畫高 = 不利
    d["exit_slip"] = sign * (planned_exit - d["exit_price"])

    risk = (d["planned_entry_price"] - d["stop_loss"]).abs()
    d["entry_bps"] = d["entry_slip"] / d["planned_entry_price"] * 1e4
    d["exit_bps"] = d["exit_slip"] / d["planned_entry_price"] * 1e4
    d["slip_r"] = (d["entry_slip"] + d["exit_slip"]) / risk

    if "entry_delay_min" in d.columns and d["entry_delay_min"].notna().any():
        dl = d["entry_delay_min"].dropna()
        print(f"\n【進場延遲】（成交時間 - 訊號K線收盤時間，{len(dl)} 筆）平均 {dl.mean():.1f} 分鐘 | 最大 {dl.max():.1f} 分鐘")
        print("  正常應該只有幾秒到一兩分鐘；偏大代表訊號被發現得晚，進場價會跟計畫價差很遠。")
    print("\n【滑價】（正值 = 吃虧；bps = 萬分之一，1 bps = 0.01%）")
    print(f"  樣本: {len(d)} 筆")
    print(f"  進場滑價  平均 {d['entry_bps'].mean():+.2f} bps | 中位數 {d['entry_bps'].median():+.2f} | 最差 {d['entry_bps'].max():+.2f}")
    print(f"  出場滑價  平均 {d['exit_bps'].mean():+.2f} bps | 中位數 {d['exit_bps'].median():+.2f} | 最差 {d['exit_bps'].max():+.2f}")
    print(f"  來回合計  平均 {(d['entry_bps'] + d['exit_bps']).mean():+.2f} bps = 每筆吃掉 {d['slip_r'].mean():+.3f} R")
    print("  （回測假設滑價為 0。0.4% 停損時，來回滑價約 8 bps 就會把淨優勢吃光，見對話說明。）")

    f = d.dropna(subset=["r_multiple", "r_multiple_after_fee"]).copy()
    if not f.empty:
        fee_r = f["r_multiple"] - f["r_multiple_after_fee"]
        risk_f = (f["entry_price"] - f["stop_loss"]).abs()
        implied_side = fee_r * risk_f / (f["entry_price"] + f["exit_price"])
        print("\n【手續費校正】")
        print(f"  實盤每筆手續費平均 {fee_r.mean():.3f} R")
        print(f"  由此反推的單邊費率約 {implied_side.mean() * 100:.4f}%（回測目前假設 {TAKER_FEE_REF * 100:.3f}%）")
        print("  → 如果差很多，請改 backtest_snr.py 的 TAKER_FEE_PCT。")


# ---------------------------------------------------------
# 主程式
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="snr_live_trades.csv")
    ap.add_argument("--backtest", default="snr_backtest_trades.csv")
    ap.add_argument("--tolerance", type=float, default=10, help="訊號時間容許差（分鐘），預設 10")
    ap.add_argument("--start", default=None, help="對帳起點（UTC），應該等於機器人最後一次啟動時間")
    ap.add_argument("--end", default=None, help="對帳終點（UTC），預設為實盤最後一筆出場時間")
    args = ap.parse_args()

    for p in (args.live, args.backtest):
        if not os.path.exists(p):
            print(f"找不到檔案: {p}")
            return

    live = load_live(args.live)
    bt = load_backtest(args.backtest)

    live_ts = live["signal_time"].dropna()
    if args.start:
        start = pd.Timestamp(args.start)
    elif not live_ts.empty:
        start = live_ts.min()
        print("⚠ 沒有指定 --start：用實盤第一筆訊號時間當起點，所以「實盤啟動後、第一筆實盤交易前」"
              "回測有交易但實盤沒有的情況會被漏掉。建議指定 --start。\n")
    else:
        print("實盤紀錄裡沒有任何帶 signal_time 的交易，無法比對。")
        return
    end = pd.Timestamp(args.end) if args.end else live["exit_time"].max()

    bt_win = bt[(bt["entry_time"] >= start) & (bt["entry_time"] <= end)].reset_index(drop=True)
    live_win = live[(live["signal_time"].isna()) | ((live["signal_time"] >= start) & (live["signal_time"] <= end))].reset_index(drop=True)

    print(f"對帳範圍(UTC): {start} → {end}")
    print(f"實盤 {len(live_win)} 筆 | 回測 {len(bt_win)} 筆 | 時間容許差 {args.tolerance:g} 分鐘\n")

    pairs, used_l, used_b = match_trades(live_win, bt_win, args.tolerance)

    rows = []
    for i, j, d in pairs:
        lr, br = live_win.loc[i], bt_win.loc[j]
        note = []
        if d > 0:
            note.append(f"訊號時間差 {d:g} 分鐘")
        px = (lr["planned_entry_price"] - br["entry_price"]) / br["entry_price"] * 100 if pd.notna(lr["planned_entry_price"]) else np.nan
        if pd.notna(px) and abs(px) > 0.02:
            note.append(f"進場價差 {px:+.3f}%（資料來源價差？）")
        same_result = lr["result"] == br["result"]
        if not same_result:
            note.append("輸贏不同")
        rows.append({
            "狀態": "一致" if d == 0 and same_result else "配對成功(有差異)" if note else "一致",
            "時間(UTC)": br["entry_time"], "方向": br["side"],
            "回測結果": br["result"], "實盤結果": lr["result"],
            "回測出場": br["exit_time"], "實盤出場": lr["exit_time"],
            "說明": "；".join(note),
        })

    bt_starts, bt_ends = bt_win["entry_time"], bt_win["exit_time"]
    lv_starts, lv_ends = live_win["signal_time"], live_win["exit_time"]

    for i, lr in live_win.iterrows():
        if i in used_l:
            continue
        if pd.isna(lr["signal_time"]):
            rows.append({"狀態": "無法比對", "時間(UTC)": lr["exit_time"], "方向": lr["side"],
                         "回測結果": "", "實盤結果": lr["result"], "回測出場": pd.NaT, "實盤出場": lr["exit_time"],
                         "說明": "重啟後接手的部位，沒有訊號時間"})
            continue
        if in_position(lr["signal_time"], bt_starts, bt_ends):
            why = "回測當時正在持倉中 → 路徑分歧（前面某筆不同的連鎖結果）"
        else:
            why = "回測當時空手卻沒有這筆 → 實盤看到了回測不認的訊號（資料或邏輯差異，最需要查）"
        rows.append({"狀態": "只有實盤", "時間(UTC)": lr["signal_time"], "方向": lr["side"],
                     "回測結果": "", "實盤結果": lr["result"], "回測出場": pd.NaT, "實盤出場": lr["exit_time"], "說明": why})

    for j, br in bt_win.iterrows():
        if j in used_b:
            continue
        if in_position(br["entry_time"], lv_starts, lv_ends):
            why = "實盤當時正在持倉中 → 路徑分歧（連鎖結果）"
        else:
            why = "實盤當時空手卻沒進場 → 查 log：離線？重啟門檻？下單失敗？緊急平倉？"
        rows.append({"狀態": "只有回測", "時間(UTC)": br["entry_time"], "方向": br["side"],
                     "回測結果": br["result"], "實盤結果": "", "回測出場": br["exit_time"], "實盤出場": pd.NaT, "說明": why})

    report = pd.DataFrame(rows).sort_values("時間(UTC)").reset_index(drop=True)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(report[["狀態", "時間(UTC)", "方向", "回測結果", "實盤結果", "說明"]].to_string(index=False))
    report.to_csv("snr_compare_report.csv", index=False, encoding="utf-8-sig")

    n_ok = (report["狀態"] == "一致").sum()
    n_diff = (report["狀態"] == "配對成功(有差異)").sum()
    n_l = (report["狀態"] == "只有實盤").sum()
    n_b = (report["狀態"] == "只有回測").sum()
    n_na = (report["狀態"] == "無法比對").sum()
    print("\n" + "=" * 60)
    print(f"完全一致 {n_ok} | 配對成功但有差異 {n_diff} | 只有實盤 {n_l} | 只有回測 {n_b} | 無法比對 {n_na}")
    bad = report[report["狀態"].isin(["只有實盤", "只有回測"])]
    if not bad.empty:
        first = bad.iloc[0]
        print(f"第一個分歧: {first['時間(UTC)']}（{first['狀態']}）→ {first['說明']}")
        print("之後的不一致很可能是這一筆的連鎖結果，先查第一個就好。")
    else:
        print("沒有任何只出現在單邊的交易。")
    print("=" * 60)

    slippage_and_fee_report(live_win)
    print("\n完整對帳表已存成 snr_compare_report.csv")


if __name__ == "__main__":
    main()