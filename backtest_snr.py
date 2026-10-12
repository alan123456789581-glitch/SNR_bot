"""
backtest_snr.py

SNR 策略回測引擎。用現貨公開資料（不需要 API key）回測，
邏輯跟 snr_trading_bot.py 一致：同時間只能有一筆交易。

使用方式：
    python backtest_snr.py

需要本機有網路能連到幣安 API（這支程式沒辦法在我這邊的沙盒環境執行，
因為這裡連不到交易所，所以你需要自己跑一次）。
"""

import time
import logging
import ccxt
import pandas as pd
import numpy as np

from snr_strategy import find_multi_timeframe_setup, RR_RATIO, LIVE_H4_LOOKBACK, LIVE_M5_LOOKBACK

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# 設定
# ---------------------------------------------------------
SYMBOL = "BTC/USDT:USDT"    # 跟正式機器人交易同一個商品（永續合約），公開資料不需要 key
TIMEFRAME_H4 = "4h"
TIMEFRAME_M5 = "5m"
BACKTEST_DAYS = 365            # 回測天數，5分鐘資料量大，建議至少 50 天以上，
                               # 確保 FRACTAL_LEFT/RIGHT 需要的「暖身資料」足夠，
                               # 不然資料視窗開頭的轉折點可能判斷不準（見對話紀錄）
# 跟實盤對帳用：填入實盤機器人「最後一次啟動」的時間（換算成 UTC，台灣時間要減8小時），
# 例如 "2026-10-09 02:00:00"，回測就只模擬從這個時間點開始的交易。
# 平常跑完整歷史回測時保持 None。注意：實盤機器人中途有重啟的話，每次重啟後的路徑
# 都不同，只能對「最後一次啟動之後、沒有中斷」的這段。
COMPARE_START_UTC = None
M5_PERIOD = pd.Timedelta(minutes=5)
H4_PERIOD = pd.Timedelta(hours=4)
# 單邊吃單手續費。進場(市價)、停損(STOP_MARKET)、停利(TAKE_PROFIT_MARKET) 全部都是吃單。
# 標準等級我記得約 0.05%（用 BNB 折抵約 0.045%），VIP 等級更低——請以你帳戶實際費率為準，
# 實盤的 snr_live_trades.csv 會記錄實際付出的手續費，可以拿來校正這個數字。
TAKER_FEE_PCT = 0.0004
# 注意：RR_RATIO 不在這裡定義，是從 snr_strategy.py 匯入的，那裡才是唯一的
# 真相來源。如果要調整盈虧比，去改 snr_strategy.py 最上面那個 RR_RATIO，
# 這支回測程式跟正式機器人會自動一起套用新的值，不用兩邊分別改。


# ---------------------------------------------------------
# 分批抓取完整歷史資料（單次 API 呼叫有根數上限，要翻頁抓）
# ---------------------------------------------------------
def fetch_full_history(exchange, symbol, timeframe, since_ms, until_ms=None):
    all_candles = []
    current_since = since_ms

    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=current_since, limit=1000)
        if not batch:
            break

        all_candles.extend(batch)
        last_timestamp = batch[-1][0]

        if until_ms and last_timestamp >= until_ms:
            break
        if len(batch) < 1000:
            break  # 已經抓到最新資料，沒有更多了

        current_since = last_timestamp + 1
        time.sleep(exchange.rateLimit / 1000)  # 尊重交易所限速

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.drop_duplicates(subset="timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)

    if until_ms:
        until_dt = pd.to_datetime(until_ms, unit="ms")
        df = df[df.index <= until_dt]

    return df


# ---------------------------------------------------------
# 模擬單筆交易：從進場點開始，逐根5分鐘K線檢查停損/停利誰先觸發
# ---------------------------------------------------------
def simulate_trade(df_5m, entry_index, plan):
    """
    從 entry_index 的下一根K線開始檢查（假設在訊號確認K線收盤時進場，
    下一根才開始承受價格波動）。

    回傳一個 dict，包含出場時間、出場價、R倍數、是贏是輸。
    同根K線同時碰到停損跟停利時，採保守假設：停損優先。
    """
    n = len(df_5m)
    signal = plan["signal"]

    for i in range(entry_index + 1, n):
        row = df_5m.iloc[i]
        high, low = row["high"], row["low"]

        if signal == "long":
            hit_sl = low <= plan["stop_loss"]
            hit_tp = high >= plan["take_profit"]
        else:
            hit_sl = high >= plan["stop_loss"]
            hit_tp = low <= plan["take_profit"]

        if hit_sl:
            return {
                "exit_time": df_5m.index[i],
                "exit_price": plan["stop_loss"],
                "result": "loss",
                "r_multiple": -1.0,
            }
        if hit_tp:
            return {
                "exit_time": df_5m.index[i],
                "exit_price": plan["take_profit"],
                "result": "win",
                # 直接讀 plan 自己算出來的 rr_ratio，不要另外用模組常數 RR_RATIO，
                # 避免兩個地方各自維護同一個數字、以後改參數時忘記同步而悄悄對不起來
                "r_multiple": plan["rr_ratio"],
            }

    # 一直到資料結尾都沒觸發停損或停利，視為尚未平倉（排除在統計外）
    return None


# ---------------------------------------------------------
# 主回測邏輯
# ---------------------------------------------------------
def run_backtest(df_4h, df_5m, left=None, right=None, start_time=None, min_stop_distance_pct=None, lookback=None,
                 h4_proximity_pct=None, h4_bias_expiry_hours=None):
    """
    直接呼叫 snr_strategy.find_multi_timeframe_setup()——跟 snr_trading_bot.py
    用的是同一個函式，不是另外維護一套平行邏輯。

    做法是忠實模擬機器人實際的輪詢過程：逐根5分鐘K線往前推進「現在」這個
    時間點，每次空手時都只用「當下能看到的資料視窗」（4H最近
    LIVE_H4_LOOKBACK根、5m最近LIVE_M5_LOOKBACK根，完全比照機器人
    fetch_ohlcv_df 的 limit）去呼叫 find_multi_timeframe_setup()。找到訊號
    就進場、模擬到出場為止，再從出場後繼續往前推進尋找下一個訊號。

    這樣寫比先前「直接走訪全部歷史4H訊號」的版本慢很多（90天大概要跑數
    分鐘，資料更長會更久），但從根本上保證回測跟正式機器人的判斷邏輯
    100%一致——不用再靠人工比對兩邊程式碼是否同步，這本身就是先前好幾輪
    對話裡不斷抓到不一致的根源。

    left/right 留 None 時，使用 find_multi_timeframe_setup() 自己的預設值
    （也就是 snr_strategy.FRACTAL_LEFT/RIGHT）。要在程式執行期間動態切換
    參數值做掃描比較（例如 backtest_sensitivity.py），透過這兩個參數傳入。
    """
    kwargs = {}
    if h4_bias_expiry_hours is not None:
        kwargs["h4_bias_expiry_hours"] = h4_bias_expiry_hours
    if h4_proximity_pct is not None:
        kwargs["h4_proximity_pct"] = h4_proximity_pct
    if lookback is not None:
        kwargs["lookback"] = lookback
    if min_stop_distance_pct is not None:
        kwargs["min_stop_distance_pct"] = min_stop_distance_pct
    if left is not None:
        kwargs["left"] = left
    if right is not None:
        kwargs["right"] = right

    trades = []
    scan_floor_time = None
    i = 0
    n = len(df_5m)

    # start_time（UTC，tz-naive）：模擬「機器人在這個時間點啟動」的狀態——從這一刻
    # 開始掃描，而且啟動時間以前的舊訊號一律不接受（跟機器人啟動時把 scan_floor_time
    # 設成「現在」的行為一致）。用來跟實盤對帳：實盤機器人是中途啟動的，回測如果
    # 從資料開頭一路模擬下來，兩邊的「目前有沒有部位、上一筆什麼時候結束」會完全
    # 不同，之後每一筆交易都會連鎖地對不上。
    if start_time is not None:
        start_time = pd.Timestamp(start_time)
        scan_floor_time = start_time
        i = int(df_5m.index.searchsorted(start_time, side="left"))

    while i < n:
        current_time = df_5m.index[i]

        # 比照機器人的效果：只給「當下」能看到的最近 N 根「已收盤」K線。
        # 這一步的「現在」= 第 i 根5分鐘K線收盤的那一刻。
        # 關鍵：4H K線必須「已經收盤」才能用。先前的寫法是取「開盤時間 <= 現在」
        # 的4H K線，等於把還在形成中、當下根本不知道最終高低收的那根4H K線，
        # 直接用它完整的歷史資料——這是偷看未來（最多偷看將近4小時）。
        now = current_time + M5_PERIOD
        h4_closed = df_4h.index.searchsorted(now - H4_PERIOD, side="right")  # 已收盤的4H根數
        if h4_closed == 0:
            i += 1
            continue  # 還沒有任何一根收盤的4H K線，跳過

        df_4h_window = df_4h.iloc[max(0, h4_closed - LIVE_H4_LOOKBACK): h4_closed]
        df_5m_window = df_5m.iloc[max(0, i - LIVE_M5_LOOKBACK + 1): i + 1]  # 第 i 根在 now 剛好收盤

        plan = find_multi_timeframe_setup(df_4h_window, df_5m_window, min_5m_time=scan_floor_time, **kwargs)

        if plan is None:
            i += 1
            continue

        entry_time = plan["entry_time"]
        full_entry_index = df_5m.index.get_loc(entry_time)
        outcome = simulate_trade(df_5m, full_entry_index, plan)

        if outcome is None:
            # 資料結尾前還沒分出勝負，不算進統計裡，但印出來讓你知道這筆
            # 交易存在，不是訊號沒抓到——不然這種「尚未平倉」的交易會悄悄
            # 消失，容易誤以為回測漏抓了某個訊號
            logger.info(
                f"（尚未平倉，不計入統計）{plan['signal']} | 進場 {entry_time} @ {plan['entry_price']:.2f} | "
                f"停損 {plan['stop_loss']:.2f} | 停利 {plan['take_profit']:.2f}"
            )
            break  # 已經到資料結尾，後面不會再有新資料了

        trade_record = {
            "h4_bias": plan["bias_from_4h"],
            "entry_time": entry_time,
            "entry_price": plan["entry_price"],
            "stop_loss": plan["stop_loss"],
            "take_profit": plan["take_profit"],
            # 訊號K線發生的時間 → 策略實際「看得到」這個訊號的時間，中間差幾分鐘。
            # 因果性修正之後應該是 0（最多一根K線）；舊邏輯下常常是好幾十分鐘甚至好幾小時，
            # 代表回測假設的進場價其實是過去的價格。
            "discovery_lag_min": (current_time - entry_time).total_seconds() / 60,
            "h4_zone_distance_pct": plan["h4_zone_distance_pct"],
            "h4_bias_age_hours": plan["h4_bias_age_hours"],
            **outcome,
        }
        trades.append(trade_record)
        scan_floor_time = outcome["exit_time"]

        logger.info(
            f"交易 #{len(trades)}: {plan['signal']} | 進場 {entry_time} @ {plan['entry_price']:.2f} | "
            f"結果 {outcome['result']} | 出場 {outcome['exit_time']}"
        )

        # 直接跳到出場時間之後：持倉期間機器人不會掃描新訊號，回測也不用
        i = df_5m.index.get_loc(outcome["exit_time"]) + 1

    return trades


# ---------------------------------------------------------
# 績效統計
# ---------------------------------------------------------
def add_fee_adjusted_r(df):
    """
    逐筆扣除手續費：來回手續費（佔進場金額的比例）÷ 這筆的停損距離（佔進場價的比例）。
    例：單邊 0.05%、停損 0.3% → 每筆手續費 = 0.10% ÷ 0.3% = 0.33R。
    停損越近，手續費佔風險的比例越大——這是這個策略（停損固定很窄）最大的成本。
    （先前的算法 (fee*2)/(1/RR) 只得到 0.0016R，等於幾乎沒有扣手續費，是錯的。）
    沒有計入：滑價（市價停損會有）、資金費率。
    """
    risk_pct = (df["entry_price"] - df["stop_loss"]).abs() / df["entry_price"]
    df["fee_in_r"] = (2 * TAKER_FEE_PCT) / risk_pct
    df["r_multiple_after_fee"] = df["r_multiple"] - df["fee_in_r"]
    return df


def snr_strategy_proximity_label():
    import snr_strategy
    v = snr_strategy.H4_ZONE_PROXIMITY_PCT
    return "無限制" if v is None else f"{v:.2%}"


def performance_report(trades):
    if not trades:
        print("沒有產生任何完整交易，無法統計績效。")
        return

    df = pd.DataFrame(trades)
    total = len(df)
    wins = (df["result"] == "win").sum()
    win_rate = wins / total

    df = add_fee_adjusted_r(df)
    if "discovery_lag_min" in df.columns:
        lag = df["discovery_lag_min"]
        print(f"訊號發現延遲：平均 {lag.mean():.1f} 分鐘 | 最大 {lag.max():.0f} 分鐘 | 延遲超過一根K線(5分鐘)的交易佔 {(lag > 5).mean():.1%}")

    df["cumulative_r"] = df["r_multiple_after_fee"].cumsum()
    running_max = df["cumulative_r"].cummax()
    drawdown = df["cumulative_r"] - running_max
    max_drawdown = drawdown.min()

    expectancy = df["r_multiple_after_fee"].mean()

    print("=" * 50)
    print(f"總交易筆數: {total}")
    print(f"勝率: {win_rate:.1%}")
    rr = df["r_multiple"].max()  # 贏單的 R（= 盈虧比）
    fee_r = df["fee_in_r"].mean()
    print(f"打平勝率（未計手續費）: {1 / (1 + rr):.1%}")
    print(f"打平勝率（含手續費）: {(1 + fee_r) / (1 + rr):.1%}   ← 實際勝率要明顯高過這個數字才有利潤")
    print(f"平均每筆毛 R（未扣手續費）: {df['r_multiple'].mean():.3f}")
    print(f"平均每筆手續費: {df['fee_in_r'].mean():.3f} R（單邊費率 {TAKER_FEE_PCT:.3%}）")
    print(f"平均每筆 R 倍數（已扣手續費，未計滑價）: {expectancy:.3f}")
    print(f"累積 R 倍數: {df['cumulative_r'].iloc[-1]:.2f}")
    print(f"最大回落（R）: {max_drawdown:.2f}")
    print("=" * 50)

    # 事先定好的成功標準（見 experiments.md）：看到結果之前就固定，不能事後改標準。
    net = df["r_multiple_after_fee"]
    t_stat = net.mean() / (net.std(ddof=1) / total ** 0.5) if total > 1 else float("nan")
    et = pd.to_datetime(df["entry_time"])
    cut = et.min() + (et.max() - et.min()) * 0.7
    early, late = net[et <= cut], net[et > cut]
    c1 = net.mean() > 0 and t_stat >= 2
    c2 = (len(early) > 0 and early.mean() > 0) and (len(late) > 0 and late.mean() > 0)
    c3 = total >= 300
    print("成功標準檢查（三項都要 ✅ 才算支持假設）:")
    print(f"  {'✅' if c1 else '❌'} 1. 淨均R > 0 且 t ≥ 2     淨均R {net.mean():+.3f}，t = {t_stat:+.2f}")
    print(f"  {'✅' if c2 else '❌'} 2. 前70%、後30%淨均R都 > 0  前 {early.mean():+.3f}（{len(early)}筆） / 後 {late.mean():+.3f}（{len(late)}筆）")
    print(f"  {'✅' if c3 else '❌'} 3. 交易筆數 ≥ 300          {total} 筆")
    if "h4_bias_age_hours" in df.columns:
        import snr_strategy
        ex = snr_strategy.H4_BIAS_EXPIRY_HOURS
        a = df["h4_bias_age_hours"]
        print(f"  （進場時 4H 方向已確立多久：平均 {a.mean():.1f} 小時，最大 {a.max():.1f} 小時；"
              f"有效期限 = {'無限制' if ex is None else f'{ex} 小時'}）")
    if "h4_zone_distance_pct" in df.columns:
        d = df["h4_zone_distance_pct"]
        print(f"  （進場價離 4H 區間的距離：平均 {d.mean():.3%}，最大 {d.max():.3%}；位置限制 = {snr_strategy_proximity_label()}）")
    print("=" * 50)

    # 月度拆解：看績效是不是每個月都撐得住，還是靠少數幾個月撐起整體數字。
    # 整段回測的總數字會把行情好壞混在一起，月度表才看得出近期有沒有轉弱。
    df["月份"] = pd.to_datetime(df["entry_time"]).dt.to_period("M")
    monthly = df.groupby("月份").agg(
        筆數=("result", "size"),
        勝率=("result", lambda x: (x == "win").mean()),
        毛R=("r_multiple", "sum"),
        淨R=("r_multiple_after_fee", "sum"),
    )
    monthly["勝率"] = monthly["勝率"].map(lambda v: f"{v:.1%}")
    print("月度績效（淨R已扣手續費、未計滑價）:")
    print(monthly.round(1).to_string())
    print("=" * 50)
    df = df.drop(columns=["月份"])

    return df


def build_data_exchange():
    """回測用的資料來源：跟機器人交易同一個商品（USDT 本位永續合約）的公開K線。
    先前用現貨資料，現貨跟永續合約有價差（基差），K線的最高/最低/收盤價會差幾十美元，
    對「影線有沒有碰到區間」這種精確判斷可能造成訊號出現與否的差異。"""
    return ccxt.binanceusdm({"enableRateLimit": True})


if __name__ == "__main__":
    exchange = build_data_exchange()

    until_ms = exchange.milliseconds()
    since_ms = until_ms - BACKTEST_DAYS * 24 * 60 * 60 * 1000

    logger.info(f"抓取 4H 資料，過去 {BACKTEST_DAYS} 天...")
    df_4h = fetch_full_history(exchange, SYMBOL, TIMEFRAME_H4, since_ms, until_ms)

    logger.info(f"抓取 5m 資料，過去 {BACKTEST_DAYS} 天...（資料量較大，需要一點時間）")
    df_5m = fetch_full_history(exchange, SYMBOL, TIMEFRAME_M5, since_ms, until_ms)

    logger.info(f"4H 資料筆數: {len(df_4h)} | 5m 資料筆數: {len(df_5m)}")

    trades = run_backtest(df_4h, df_5m, start_time=COMPARE_START_UTC)
    result_df = performance_report(trades)

    if result_df is not None:
        result_df.to_csv("snr_backtest_trades.csv", index=False)
        logger.info("交易明細已存成 snr_backtest_trades.csv")