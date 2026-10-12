"""
snr_strategy.py

多重時間框架 SNR (Support and Resistance) 策略
- 4H 找到 regular 或 breakout(flipped SNR)，得到方向（long / short）
- 5m 再找「同方向」的 regular 或 breakout，出現就進場
- 頭肩型態已依指示移除

區間（ZONE_MODE = "pair_peak"，照圖片的畫法）：
  壓力 = 相鄰兩根K線「漲K接跌K」，且這對K線的最高點是前面 SWING_LOOKBACK 根K線裡最高的（明顯高點）
  支撐 = 相鄰兩根K線「跌K接漲K」，且這對K線的最低點是前面 SWING_LOOKBACK 根K線裡最低的（明顯低點）
  區間 = 這兩根K線的實體邊緣（收盤價與下一根開盤價）之間的細帶，第二根K線收盤時就成立。
  「明顯」只看左邊（過去），不看未來，所以訊號不會偷看後面的K線。

regular：區間從沒被碰過，第一次只有影線碰到、實體留在原本那一側 → 守住（支撐 long／壓力 short）
breakout：K線收盤穿越區間（角色翻轉），之後第一次只有影線碰回、實體留在新的一側 → 延續突破方向
（兩者的詳細判斷見 evaluate_zone）

ZONE_MODE = "fractal" 是舊的轉折點畫法（要等後面 8 根K線確認，有時間延遲），僅供對照。
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum


# ===========================================================
# 策略參數：全系統唯一的真相來源
# backtest_snr.py 跟 snr_trading_bot.py 都應該從這裡讀取 RR_RATIO，
# 不要各自在自己的檔案裡再定義一份同名或同值的常數，避免以後調整參數時
# 改了一邊忘了改另一邊，導致回測結果跟實際機器人用的比例悄悄對不起來
# ===========================================================
RR_RATIO = 2.0   # 盈虧比 1:2

# 最小停損距離（佔進場價的百分比）。確認K線的收盤價有時會剛好貼著區間邊緣，
# 算出來的停損距離小到不切實際——這種距離在真實下單時，從「市價進場成交」
# 到「掛停損單」這幾百毫秒的正常價格波動，就足以讓交易所拒絕這張停損單
# （錯誤碼 -2021 "Order would immediately trigger"）。回測沒辦法模擬這個
# 執行瞬間的時間差，所以這類交易在回測裡會被正常計入，但實盤會頻繁觸發
# 緊急平倉、白白吃掉手續費。
#
# 這個數字沒辦法純靠理論算出來，只能靠 Demo Trading 實測校準：
#   0.0015（0.15%）實測後仍然遇到 -2021 錯誤，代表不夠寬，已調高到 0.003。
#   如果之後還是偶爾遇到同樣的錯誤，代表還要再往上調；如果長時間都不再
#   遇到，可以考慮往下微調，換取更緊的停損、更高的資金效率。
MIN_STOP_DISTANCE_PCT = 0.004

# fractal 轉折偵測的左右比較根數。數字越大，抓到的轉折點越少但越可靠
# （因為要比較更多根K線），數字越小則相反。這也是全系統唯一真相來源：
# 要調整這個參數，只改這裡，不要去改 find_pivots() 或 scan_timeframe()
# 函式簽名裡的預設值——那些預設值只有在呼叫端完全不傳參數時才會生效，
# 本系統裡 scan_timeframe() 一定會明確傳 left/right 給 find_pivots()，
# 所以 find_pivots() 自己的預設值實際上永遠不會被用到。
FRACTAL_LEFT = 2
FRACTAL_RIGHT = 2

# 正式機器人每輪迴圈實際抓取的資料視窗大小（最近幾根K線）。backtest_snr.py
# 現在會直接模擬機器人的輪詢過程，所以也需要用一模一樣的視窗大小，否則回測
# 看到的歷史範圍會比機器人實際運行時看到的更寬或更窄，判斷結果可能對不起來。
LIVE_H4_LOOKBACK = 200
LIVE_M5_LOOKBACK = 500

# ---- 訊號的「因果性」設定（實盤與回測共用）----
# 問題背景：fractal 轉折點要等後面再走 right 根K線才能確認。但 evaluate_zone() 是從轉折點的
# 下一根K線開始找「第一次碰到區間」，所以現在才確認出來的區間，可能早在 right 根K線以前就已經
# 被碰過了。若直接拿這個過去的訊號進場，實盤會用「現在的市價」去追「過去那根K線」的訊號
# （進場價跟計畫價差很遠，停損停利距離跟 R 倍數全部走樣），回測則更樂觀：直接假設在過去
# 那根K線的收盤價成交，還用事後才知道的價格路徑去模擬。
#
# REQUIRE_ZONE_KNOWN：訊號的確認K線必須發生在「區間已經被確認」之後（轉折點 + right 根）。
#                     區間被確認之前發生的碰觸，當時根本不知道區間存在，不能進場
#                     （但仍然算「已被碰過」，區間不再產生之後的訊號）。
# MAX_SIGNAL_AGE_CANDLES：5m 訊號的確認K線，距離「最新一根已收盤K線」最多幾根。1 = 只接受剛剛
#                     收盤的那根（最多再容許晚 1 根，應付輪詢稍微晚一點）。None = 不限制（舊行為）。
# 想重現舊邏輯做對照：REQUIRE_ZONE_KNOWN = False、MAX_SIGNAL_AGE_CANDLES = None。
# 這兩個值在函式「執行時」才讀取（不是預設參數），所以可以在執行期間動態切換做 A/B 比較。
REQUIRE_ZONE_KNOWN = True
MAX_SIGNAL_AGE_CANDLES = 1

# ---- 區間的畫法 ----
# "pair_peak"：圖片的畫法（漲接跌／跌接漲 + 明顯高低點），區間在第二根K線收盤時就成立。
# "fractal"  ：舊的轉折點畫法，要等後面再走 right 根K線才確認，僅供對照。
ZONE_MODE = "pair_peak"
# pair_peak 模式的「明顯」門檻：這對K線的高點（低點）要比前面幾根K線都更高（更低）。
# 數字越大 = 只留下更重要的高低點、區間更少。這個值沒有標準答案，回測用
# backtest_sensitivity.py 掃描，並且要用「沒拿來挑參數的那一段資料」驗證，不要只看全期績效。
SWING_LOOKBACK = 20

# ---- 假設一（正在驗證）：5M 進場必須在「產生 4H 方向的那個 4H 區間」附近 ----
# 進場K線的收盤價，距離該 4H 區間（區間內算 0）不能超過這個比例。0.4% = 目前的停損距離，意思是
# 「價格離區間不超過一個風險單位」。這個值是事先固定的，不要因為看過結果就回頭調它。
# None = 不限制位置（先前的版本）。判斷標準與結果記錄在 experiments.md。
H4_ZONE_PROXIMITY_PCT = None   # 假設一已被否定（見 experiments.md），預設關閉；要重現實驗請改回 0.004

# ---- 假設二（正在驗證）：4H 方向只在確立後的一段時間內有效 ----
# 4H 方向確立（那根 4H K線收盤）之後，超過這麼多小時就視為過期，不再用它的方向進場。
# 目前的做法是「一直有效，直到出現更新的 4H 訊號」，可能拖很久。24 小時（6 根 4H K線）是事先
# 固定的單一數值，沒有依據，不掃參數。None = 不限制（先前的版本）。判斷標準與結果記錄在 experiments.md。
H4_BIAS_EXPIRY_HOURS = 24

M5_PERIOD = pd.Timedelta(minutes=5)
H4_PERIOD = pd.Timedelta(hours=4)


# ===========================================================
# 資料結構定義
# ===========================================================
class ZoneType(Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"


class ZoneStatus(Enum):
    FRESH = "fresh"                      # 尚未被測試過
    FLIPPED = "flipped"                  # 已被實體突破/跌破，角色反轉中，等待情況二回測
    CONFIRMED_HOLD = "confirmed_hold"    # 情況一成立：守住原角色
    CONFIRMED_FLIP = "confirmed_flip"    # 情況二成立：反轉後新角色守住
    INVALIDATED = "invalidated"          # 兩種情況都不成立，區間失效


@dataclass
class Zone:
    zone_type: ZoneType
    top: float
    bottom: float
    formed_at_index: int       # 區間形成當下是第幾根K線（用來之後比對時間）
    status: ZoneStatus = ZoneStatus.FRESH
    tested_count: int = 0

    def contains(self, price):
        return self.bottom <= price <= self.top


def body_high(row):
    return max(row["open"], row["close"])


def body_low(row):
    return min(row["open"], row["close"])


# ===========================================================
# 步驟 1：找局部高低點（轉折候選）
# ===========================================================
def find_pivots(df, left=FRACTAL_LEFT, right=FRACTAL_RIGHT):
    """
    left/right 代表要比較左右各幾根K線才能算局部極值。
    數字越大，濾掉的雜訊越多，但也要等右邊 N 根K線收完才能確認（有延遲）。
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    pivot_high = np.zeros(n, dtype=bool)
    pivot_low = np.zeros(n, dtype=bool)

    for i in range(left, n - right):
        window_high = highs[i - left: i + right + 1]
        window_low = lows[i - left: i + right + 1]
        if highs[i] == window_high.max():
            pivot_high[i] = True
        if lows[i] == window_low.min():
            pivot_low[i] = True

    return pivot_high, pivot_low


# ===========================================================
# 步驟 2：把轉折K線跟前一根相鄰K線配對，畫出區間
# ===========================================================
def build_zones_from_pivots(df, pivot_high, pivot_low):
    zones = []
    n = len(df)

    for i in range(1, n):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        if pivot_high[i]:
            top = max(body_high(prev), body_high(curr))
            bottom = min(body_high(prev), body_high(curr))
            zones.append(Zone(ZoneType.RESISTANCE, top, bottom, formed_at_index=i))

        if pivot_low[i]:
            top = max(body_low(prev), body_low(curr))
            bottom = min(body_low(prev), body_low(curr))
            zones.append(Zone(ZoneType.SUPPORT, top, bottom, formed_at_index=i))

    return zones


def distance_to_zone_pct(price, zone):
    """價格到區間的距離（佔價格的比例）；價格落在區間內 = 0。"""
    if zone.bottom <= price <= zone.top:
        return 0.0
    return min(abs(price - zone.bottom), abs(price - zone.top)) / price


# ===========================================================
# 步驟 2b：圖片的畫法——相鄰兩根「漲接跌／跌接漲」 + 明顯高低點
# ===========================================================
def build_zones_from_pairs(df, lookback):
    """
    壓力：K線 k 收漲、K線 k+1 收跌（一個「峰」），且這兩根的最高點高於前面 lookback 根K線的最高點。
    支撐：K線 k 收跌、K線 k+1 收漲（一個「谷」），且這兩根的最低點低於前面 lookback 根K線的最低點。
    區間 = 收盤價（k）與開盤價（k+1）之間，也就是兩根實體邊緣對齊的位置。
    formed_at_index = k+1：第二根收盤時區間就成立，只用到過去的資料。
    """
    o = df["open"].to_numpy(); c = df["close"].to_numpy()
    h = df["high"].to_numpy(); l = df["low"].to_numpy()
    n = len(df)
    zones = []
    for k in range(lookback, n - 1):
        j = k + 1
        if c[k] > o[k] and c[j] < o[j] and max(h[k], h[j]) > h[k - lookback:k].max():
            zones.append(Zone(ZoneType.RESISTANCE, top=max(c[k], o[j]), bottom=min(c[k], o[j]), formed_at_index=j))
        if c[k] < o[k] and c[j] > o[j] and min(l[k], l[j]) < l[k - lookback:k].min():
            zones.append(Zone(ZoneType.SUPPORT, top=max(c[k], o[j]), bottom=min(c[k], o[j]), formed_at_index=j))
    return zones


# ===========================================================
# 步驟 3：逐根檢查區間後續走勢，判斷情況一/情況二是否成立
# ===========================================================
def evaluate_zone(df, zone: Zone):
    """
    從區間形成後開始，逐根檢查後續K線，判斷圖上的兩種型態（頭肩已移除）：

    regular（第一次測試）：區間從沒被碰過，之後第一次有K線「只有影線碰到區間、實體完全留在
        區間原本那一側」→ 守住原本的角色（支撐 → long，壓力 → short）。
    breakout / flipped SNR（突破後回測）：K線「收盤」穿越到區間另一側（實體穿過區間並完成），
        區間角色翻轉；之後第一次有K線「只有影線碰回區間、實體留在新的那一側」→ 延續突破方向
        （壓力被向上突破 → long，支撐被向下跌破 → short）。

    判斷細節（以支撐區間 [bottom, top] 為例，壓力區間完全鏡像）：
      - 收盤 < bottom：實體穿越並收在區間下方 → 翻轉，等待回測。穿越那一根本身不能同時當回測。
      - 影線碰到區間（low <= top）且實體完全在區間上方（實體最低點 > top）→ 乾淨的影線測試。
      - 影線碰到區間、但實體有一部分進入區間（或穿越後又收回上方）→ 不是乾淨的影線測試，
        不產生訊號；但區間已經「被碰過」，之後不再是第一次，也就不會再產生 regular 訊號。
        （先前版本會把這種K線也當成 regular 訊號，甚至包含收盤已經跌破支撐的K線，是錯的。）

    回傳 (signal, confirm_index):
        signal: 'long' / 'short' / None
        confirm_index: 訊號確認當下是第幾根K線
    """
    return _evaluate_zone_arrays(df["open"].to_numpy(), df["high"].to_numpy(),
                                 df["low"].to_numpy(), df["close"].to_numpy(), zone)


def _evaluate_zone_arrays(o, h, l, c, zone: Zone):
    """evaluate_zone 的實際實作，直接吃 numpy 陣列（比逐列 df.iloc 快很多）。"""
    n = len(c)
    start = zone.formed_at_index + 1

    for i in range(start, n):
        oi, close = o[i], c[i]
        b_high, b_low = (oi, close) if oi > close else (close, oi)
        high, low = h[i], l[i]

        if zone.status == ZoneStatus.FRESH:
            if zone.zone_type == ZoneType.SUPPORT:
                if close < zone.bottom:
                    zone.status = ZoneStatus.FLIPPED   # 實體穿越並收在下方：支撐變壓力
                    continue
                if low <= zone.top:                    # K線範圍碰到區間
                    if b_low > zone.top:               # 實體完全在區間上方：乾淨的影線測試
                        if zone.tested_count == 0:
                            zone.status = ZoneStatus.CONFIRMED_HOLD
                            return "long", i
                        zone.status = ZoneStatus.INVALIDATED   # 之前就碰過了，不是第一次
                        return None, i
                    zone.tested_count += 1             # 實體進入區間：碰過了，但不算乾淨測試

            else:  # RESISTANCE
                if close > zone.top:
                    zone.status = ZoneStatus.FLIPPED   # 實體穿越並收在上方：壓力變支撐
                    continue
                if high >= zone.bottom:
                    if b_high < zone.bottom:           # 實體完全在區間下方：乾淨的影線測試
                        if zone.tested_count == 0:
                            zone.status = ZoneStatus.CONFIRMED_HOLD
                            return "short", i
                        zone.status = ZoneStatus.INVALIDATED
                        return None, i
                    zone.tested_count += 1

        elif zone.status == ZoneStatus.FLIPPED:
            if zone.zone_type == ZoneType.SUPPORT:
                # 原支撐被跌破，現在是壓力：等影線回測、實體留在區間下方
                if high >= zone.bottom and b_high < zone.bottom:
                    zone.status = ZoneStatus.CONFIRMED_FLIP
                    return "short", i
                if close > zone.top:                   # 又收回區間上方：假跌破，失效
                    zone.status = ZoneStatus.INVALIDATED
                    return None, i

            else:  # 原壓力被突破，現在是支撐
                if low <= zone.top and b_low > zone.top:
                    zone.status = ZoneStatus.CONFIRMED_FLIP
                    return "long", i
                if close < zone.bottom:                # 又收回區間下方：假突破，失效
                    zone.status = ZoneStatus.INVALIDATED
                    return None, i

        else:
            break  # 已有結論的區間不再重複判斷

    return None, None


# ===========================================================
# 步驟 4：單一時間框架的完整掃描（找出所有區間 + 對應訊號）
# ===========================================================
def scan_timeframe(df, left=FRACTAL_LEFT, right=FRACTAL_RIGHT, lookback=None):
    """
    回傳一個 list，每個元素是 (zone, signal, confirm_index)，
    只保留有產生訊號（long/short）的區間。

    ZONE_MODE = "pair_peak"：用 lookback（預設 SWING_LOOKBACK）；left/right 不使用。
    ZONE_MODE = "fractal"  ：舊畫法，用 left/right；lookback 不使用。
    """
    if ZONE_MODE == "pair_peak":
        zones = build_zones_from_pairs(df, SWING_LOOKBACK if lookback is None else lookback)
        known_lag = 0   # 第二根K線收盤時區間就成立，沒有確認延遲
    else:
        pivot_high, pivot_low = find_pivots(df, left=left, right=right)
        zones = build_zones_from_pivots(df, pivot_high, pivot_low)
        known_lag = right

    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()

    results = []
    for zone in zones:
        signal, confirm_index = _evaluate_zone_arrays(o, h, l, c, zone)
        if signal is None:
            continue
        # fractal 舊畫法：區間要等轉折點之後再走 right 根K線才確認得出來，之前發生的碰觸不能進場
        if REQUIRE_ZONE_KNOWN and confirm_index < zone.formed_at_index + known_lag:
            continue
        results.append((zone, signal, confirm_index))
    return results


# ===========================================================
# 步驟 5：4H 偏向 + 5m 進場，組合成完整交易計畫
# ===========================================================
def calculate_trade_plan(signal, zone: Zone, confirm_row, rr_ratio=RR_RATIO, buffer_pct=0.0005,
                          min_stop_distance_pct=MIN_STOP_DISTANCE_PCT):
    """
    用 5m 確認訊號當下的K線跟區間，算出進場價、停損、停利。
    buffer_pct: 停損價再往外推一點點的緩衝，避免剛好卡在區間邊緣被掃損。

    如果區間算出來的停損距離小於 min_stop_distance_pct（實務上太近，下單
    瞬間的正常價格雜訊就可能讓停損單被交易所拒絕），不會放棄這個訊號，
    而是把停損直接撐開到最小安全距離，停利依然用同樣的 rr_ratio 等比例
    跟著拉遠，維持原本的盈虧比不變。這樣不會改變「交易了哪些訊號」，
    只是讓價位更貼近真實下單能夠執行的樣子。

    （先前版本是「停損太近就整個放棄這個訊號、改找下一個候選」，實測後
    發現這樣反而會把一些測試精準、品質可能更好的訊號換成別的訊號，
    拉低了整體表現，所以改成現在這種「撐開距離、保留訊號」的做法。）
    """
    entry_price = confirm_row["close"]

    if signal == "long":
        zone_based_stop = zone.bottom * (1 - buffer_pct)
        min_distance_stop = entry_price * (1 - min_stop_distance_pct)
        stop_loss = min(zone_based_stop, min_distance_stop)  # 取離進場價較遠的那個
        risk = entry_price - stop_loss
        take_profit = entry_price + risk * rr_ratio
    else:  # short
        zone_based_stop = zone.top * (1 + buffer_pct)
        min_distance_stop = entry_price * (1 + min_stop_distance_pct)
        stop_loss = max(zone_based_stop, min_distance_stop)
        risk = stop_loss - entry_price
        take_profit = entry_price - risk * rr_ratio

    if risk <= 0:
        return None  # 理論上不會發生，這裡只是防呆

    return {
        "signal": signal,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_per_unit": risk,
        "rr_ratio": rr_ratio,
    }


def find_multi_timeframe_setup(df_4h, df_5m, left=FRACTAL_LEFT, right=FRACTAL_RIGHT, min_5m_time=None,
                               min_stop_distance_pct=None, lookback=None, h4_proximity_pct=None, h4_bias_expiry_hours=None):
    """
    先在 4H 找出方向（regular 或 breakout），再在 5m 找「同方向」的 regular 或 breakout 當進場點。

    1. 4H 方向：取「最新確認」的 4H 訊號。這個方向要等那根 4H K線「收盤」才確立
       （確認K線的開盤時間 + 4 小時）。
    2. 5m 訊號：區間用完整的 5m 資料來判斷（「明顯高低點」要看得到前面的K線），但**進場訊號**
       的確認K線必須在 4H 方向確立之後，而且晚於 min_5m_time（上一筆交易結束／機器人啟動的時間）。
    3. 訊號必須是剛剛才發生的（MAX_SIGNAL_AGE_CANDLES），過期的不追。
    4. （假設一）進場價必須在產生 4H 方向的那個 4H 區間附近（H4_ZONE_PROXIMITY_PCT）。
    5. （假設二）4H 方向確立後超過 H4_BIAS_EXPIRY_HOURS 小時就過期。
    6. 依時間先後依序嘗試符合的訊號，取第一個停損距離、位置、方向有效期都合格的。

    min_5m_time: 只接受這個時間之後的 5m 訊號（排除已經交易過、或持倉／離線期間出現的舊訊號）。
    h4_proximity_pct: 留 None = 用 H4_ZONE_PROXIMITY_PCT；傳 float('inf') = 明確關閉位置限制（做 A/B 對照用）。
    h4_bias_expiry_hours: 留 None = 用 H4_BIAS_EXPIRY_HOURS；傳 float('inf') = 明確關閉有效期限（做 A/B 對照用）。
    min_stop_distance_pct / lookback: 留 None = 用模組預設值（實盤就是這樣）；只有回測做參數掃描
                 時才明確傳入。不要靠改模組變數：函式預設值在載入時就綁定了。
    """
    scan_kwargs = {"left": left, "right": right, "lookback": lookback}

    h4_results = scan_timeframe(df_4h, **scan_kwargs)
    if not h4_results:
        return None

    h4_zone, h4_signal, h4_confirm_index = sorted(h4_results, key=lambda r: r[2])[-1]
    h4_confirm_time = df_4h.index[h4_confirm_index]
    h4_close_time = h4_confirm_time + H4_PERIOD   # 這根 4H K線收盤、方向確立的時間

    m5_results = scan_timeframe(df_5m, **scan_kwargs)
    matching = sorted((r for r in m5_results if r[1] == h4_signal), key=lambda r: r[2])
    if not matching:
        return None

    n5 = len(df_5m)
    for m5_zone, m5_signal, m5_confirm_index in matching:
        entry_time = df_5m.index[m5_confirm_index]

        # 4H 方向還沒確立之前出現的 5m 訊號不算
        if entry_time + M5_PERIOD < h4_close_time:
            continue
        # 上一筆交易結束／機器人啟動之前的訊號不算
        if min_5m_time is not None and entry_time <= min_5m_time:
            continue
        # 只接受剛剛才發生的訊號：離最新一根已收盤K線太遠，代表過期，現在追進去價格已經跟計畫價差很遠
        if MAX_SIGNAL_AGE_CANDLES is not None and (n5 - 1 - m5_confirm_index) > MAX_SIGNAL_AGE_CANDLES:
            continue

        confirm_row = df_5m.iloc[m5_confirm_index]

        # 假設二：4H 方向確立後超過有效期限就不再進場
        expiry = H4_BIAS_EXPIRY_HOURS if h4_bias_expiry_hours is None else h4_bias_expiry_hours
        bias_age_hours = (entry_time + M5_PERIOD - h4_close_time).total_seconds() / 3600
        if expiry is not None and bias_age_hours > expiry:
            continue

        # 假設一：進場價必須在 4H 區間附近
        proximity_limit = H4_ZONE_PROXIMITY_PCT if h4_proximity_pct is None else h4_proximity_pct
        h4_distance = distance_to_zone_pct(confirm_row["close"], h4_zone)
        if proximity_limit is not None and h4_distance > proximity_limit:
            continue

        plan_kwargs = {} if min_stop_distance_pct is None else {"min_stop_distance_pct": min_stop_distance_pct}
        plan = calculate_trade_plan(m5_signal, m5_zone, confirm_row, **plan_kwargs)
        if plan is None:
            continue  # 停損距離太近，放棄這個訊號，試下一個

        plan["bias_from_4h"] = h4_signal
        plan["h4_zone"] = h4_zone
        plan["m5_zone"] = m5_zone
        plan["h4_confirm_time"] = h4_confirm_time
        plan["h4_zone_distance_pct"] = h4_distance
        plan["h4_bias_age_hours"] = bias_age_hours
        plan["entry_time"] = entry_time
        return plan

    return None  # 沒有符合條件的新訊號


if __name__ == "__main__":
    # 範例：假設你已經用 ccxt 抓好 4H 跟 5m 的 df（欄位需含 open/high/low/close）
    # from crypto_analysis_ccxt import fetch_ohlcv_df
    # df_4h = fetch_ohlcv_df("binance", "BTC/USDT", "4h", limit=200)
    # df_5m = fetch_ohlcv_df("binance", "BTC/USDT", "5m", limit=500)
    # plan = find_multi_timeframe_setup(df_4h, df_5m)
    # print(plan)
    pass