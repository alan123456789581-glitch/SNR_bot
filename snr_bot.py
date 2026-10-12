"""
snr_trading_bot.py

SNR 多重時間框架策略 + 幣安 USDT 本位合約下單整合

執行前必讀:
1. SANDBOX_MODE = True 先在合約測試網跑過，測試網 key 要去
   https://testnet.binancefuture.com 另外申請，跟正式帳戶 key 不同
2. 合約有槓桿，虧損速度比現貨快很多，LEVERAGE 和 MARGIN_USDT_PER_TRADE
   請先設保守的數字
3. 這支程式目前用「輪詢 + 每次重新掃描近期K線」的簡化方式判斷狀態，
   不是真正逐根K線遞增更新的狀態機，程式重啟或掃描範圍設太短都可能
   漏掉或重複判斷同一個區間。正式上線前請先用測試網觀察至少數天，
   比對 log 紀錄跟實際盤面，確認行為符合預期
"""

import os
import json
import time
import logging
import ccxt
import pandas as pd
from dotenv import load_dotenv

from snr_strategy import find_multi_timeframe_setup, LIVE_H4_LOOKBACK, LIVE_M5_LOOKBACK

# ---------------------------------------------------------
# 設定
# ---------------------------------------------------------
load_dotenv()

SANDBOX_MODE = True
SYMBOL = "BTC/USDT:USDT"       # ccxt 統一格式的合約代號，USDT本位永續合約要加 :USDT
TIMEFRAME_H4 = "4h"
TIMEFRAME_M5 = "5m"
CHECK_INTERVAL_SECONDS = 60 * 5     # 跟 5 分鐘對齊，每 5 分鐘檢查一次

LEVERAGE = 10                        # 槓桿倍數，先保守設定
MARGIN_USDT_PER_TRADE = 50          # 每筆交易投入的保證金（不是名目金額）
MARGIN_MODE = "isolated"            # isolated（逐倉）比 cross（全倉）風險較可控，初學建議逐倉

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    # 明確指定 UTF-8：FileHandler 不指定編碼時，Windows 可能用系統地區編碼
    # （例如 Big5/cp950）寫檔，之後用別的工具或編碼開啟中文字就會變亂碼
    handlers=[logging.FileHandler("snr_trading_bot.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

STATE_FILE = "snr_bot_state.json"


# ---------------------------------------------------------
# scan_floor_time 持久化：寫進檔案，讓防重複進場的機制撐過程式重啟
# ---------------------------------------------------------
def save_scan_floor_time(scan_floor_time):
    """
    每次 scan_floor_time 更新時都呼叫這個函式。不這樣做的話，程式只要
    重啟一次（不是當掉自動重啟，是你手動關掉重開），記憶體裡的
    scan_floor_time 就會消失，一個已經交易過、已經平倉的舊訊號，只要
    還留在最近的資料視窗內，就可能被重新判定成「新」訊號、重複交易一次。
    """
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(
                {"scan_floor_time": scan_floor_time.isoformat() if scan_floor_time is not None else None},
                f,
            )
    except OSError as e:
        logger.warning(f"寫入狀態檔案失敗，不影響本次運行，但下次重啟可能無法正確避開舊訊號：{e}")


def load_scan_floor_time():
    """程式啟動時讀回上次記錄的 scan_floor_time，沒有檔案或讀取失敗就回傳 None。"""
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        value = data.get("scan_floor_time")
        return pd.Timestamp(value) if value else None
    except FileNotFoundError:
        return None
    except (OSError, ValueError, json.JSONDecodeError) as e:
        logger.warning(f"讀取狀態檔案失敗，視為沒有舊紀錄：{e}")
        return None


# ---------------------------------------------------------
# 交易所連線（合約版）
# ---------------------------------------------------------
def build_exchange():
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        raise RuntimeError("找不到 API key，請確認 .env 檔案存在且變數名稱正確")

    # 合約要用 binanceusdm，不是 binance（binance 是現貨用的）
    exchange = ccxt.binanceusdm({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
    })

    if SANDBOX_MODE:
        # 注意：幣安合約的舊版 testnet（testnet.binancefuture.com）已經被官方關閉，
        # ccxt 改用新的 enable_demo_trading() 介面，對應幣安新的「Demo Trading」機制。
        # Demo Trading 的 API key 要另外去 https://demo.binance.com/en/my/settings/api-management
        # 申請，跟正式帳戶、以及舊的 testnet key 都不是同一組。
        exchange.enable_demo_trading(True)
        logger.info("目前為合約 Demo Trading 模式，不會動用真實資金")
    else:
        logger.warning("目前為正式合約環境，將會使用真實資金並帶有槓桿！")

    exchange.load_markets()

    try:
        exchange.set_margin_mode(MARGIN_MODE, SYMBOL)
    except ccxt.ExchangeError as e:
        # 如果已經是這個模式，幣安會回傳錯誤，直接略過即可
        logger.info(f"設定保證金模式略過（可能已經是 {MARGIN_MODE}）：{e}")

    exchange.set_leverage(LEVERAGE, SYMBOL)
    logger.info(f"槓桿設定為 {LEVERAGE}x，保證金模式：{MARGIN_MODE}")

    return exchange


# ---------------------------------------------------------
# 抓資料
# ---------------------------------------------------------
def fetch_ohlcv_df(exchange, symbol, timeframe, limit=300):
    """
    只回傳「已收盤」的K線。交易所的 fetch_ohlcv 最後一根永遠是「還在形成中、
    高低收還會變動」的那根K線；拿它去判斷訊號，會用到之後會改變的資料，
    也跟回測（只看已收盤K線）不一致。所以多抓一根，把最後那根未收盤的丟掉。
    """
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit + 1)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df.iloc[:-1]


# ---------------------------------------------------------
# 部位狀態追蹤
# ---------------------------------------------------------
class PositionState:
    def __init__(self):
        self.in_position = False
        self.side = None            # 'long' / 'short'
        self.entry_order_id = None
        self.sl_order_id = None
        self.tp_order_id = None
        # 下一次掃描訊號時，5m訊號必須晚於這個時間點才會被接受。
        # 兩個時機會更新這個值：
        #   1. 剛送出一筆交易時，設成這筆交易的進場時間，避免同一個訊號
        #      在下一輪迴圈被重複判定成「最早的訊號」而重複進場
        #   2. 偵測到部位已經平倉時，設成「現在的時間」，避免持倉期間
        #      曾經出現過、但機器人當時沒去理會的舊訊號，等出場後又被
        #      誤判成新訊號去交易——這點是為了跟 backtest_snr.py 用
        #      「上一筆交易的出場時間」當門檻保持一致，不是用進場時間
        # 這個狀態只存在記憶體裡，程式重啟就會遺失——是之後要優化成
        # 持久化儲存（寫檔或資料庫）的地方。
        self.scan_floor_time = None

        # 這筆交易的計畫資訊，平倉時用來寫入交易紀錄（snr_live_trades.csv）
        self.planned_entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.signal_time = None     # 5m 訊號確認K線的時間，跟回測的 entry_time 同一個概念，方便比對
        self.h4_bias = None

    def enter(self, side, entry_order_id, sl_order_id, tp_order_id):
        self.in_position = True
        self.side = side
        self.entry_order_id = entry_order_id
        self.sl_order_id = sl_order_id
        self.tp_order_id = tp_order_id

    def set_trade_details(self, planned_entry_price, stop_loss, take_profit, signal_time, h4_bias):
        self.planned_entry_price = planned_entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.signal_time = signal_time
        self.h4_bias = h4_bias

    def exit(self):
        self.in_position = False
        self.side = None
        self.entry_order_id = None
        self.sl_order_id = None
        self.tp_order_id = None
        self.planned_entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.signal_time = None
        self.h4_bias = None


# ---------------------------------------------------------
# 實盤交易紀錄（格式比照回測的 snr_backtest_trades.csv，方便長期比對）
# ---------------------------------------------------------
LIVE_TRADES_CSV = "snr_live_trades.csv"


def extract_trade_fills(exchange, symbol, side):
    """
    從最近的實際成交紀錄（fetch_my_trades，標準端點，不受 Algo Order 影響）
    反推這筆交易的進場成交與出場成交：
    從最新的成交往回找，先收集結尾連續的「平倉方向」成交，再往前收集「進場
    方向」成交，直到累積數量等於平倉數量為止（避免連續兩筆方向相反的交易時，
    前一筆的平倉成交被誤算成這筆的進場）。
    """
    close_side = "sell" if side == "long" else "buy"
    entry_side = "buy" if side == "long" else "sell"

    trades = exchange.fetch_my_trades(symbol, None, 100)
    trades = sorted(trades, key=lambda t: t["timestamp"])

    i = len(trades)
    close_fills = []
    while i > 0 and trades[i - 1].get("side") == close_side:
        close_fills.append(trades[i - 1])
        i -= 1

    need = sum(t["amount"] for t in close_fills)
    entry_fills = []
    while i > 0 and trades[i - 1].get("side") == entry_side and need > 1e-12:
        entry_fills.append(trades[i - 1])
        need -= trades[i - 1]["amount"]
        i -= 1

    return entry_fills, close_fills


def _vwap(fills):
    total = sum(t["amount"] for t in fills)
    return sum(t["price"] * t["amount"] for t in fills) / total if total > 0 else None


def _sum_fees_usdt(fills):
    total = 0.0
    for t in fills:
        fee = t.get("fee") or {}
        if fee.get("currency") in (None, "USDT") and fee.get("cost") is not None:
            total += float(fee["cost"])
    return total


def record_closed_trade(exchange, symbol, state: PositionState):
    """
    平倉後，把這筆交易的實際進出場價格、結果、R倍數寫進 CSV。
    這個函式任何失敗都只記錄警告，不會影響交易主迴圈繼續運作。
    """
    try:
        entry_fills, close_fills = extract_trade_fills(exchange, symbol, state.side)
        if not close_fills:
            logger.warning("找不到這筆交易的平倉成交紀錄，無法寫入交易紀錄")
            return

        exit_price = _vwap(close_fills)
        exit_time = pd.to_datetime(max(t["timestamp"] for t in close_fills), unit="ms")
        entry_price = _vwap(entry_fills) if entry_fills else state.planned_entry_price
        entry_fill_time = pd.to_datetime(min(t["timestamp"] for t in entry_fills), unit="ms") if entry_fills else pd.NaT
        # 進場延遲 = 實際成交時間 - 訊號K線收盤時間。對齊K線輪詢後應該只有幾秒到幾十秒
        entry_delay_min = None
        if pd.notna(entry_fill_time) and state.signal_time is not None and pd.notna(state.signal_time):
            entry_delay_min = (entry_fill_time - (pd.Timestamp(state.signal_time) + pd.Timedelta(minutes=5))).total_seconds() / 60
        amount = sum(t["amount"] for t in close_fills)
        fees = _sum_fees_usdt(entry_fills) + _sum_fees_usdt(close_fills)

        if state.side == "long":
            pnl_per_unit = exit_price - entry_price
        else:
            pnl_per_unit = entry_price - exit_price

        r_multiple = None
        r_after_fee = None
        if state.stop_loss is not None and entry_price is not None:
            risk_per_unit = abs(entry_price - state.stop_loss)
            if risk_per_unit > 0:
                r_multiple = pnl_per_unit / risk_per_unit
                r_after_fee = (pnl_per_unit * amount - fees) / (risk_per_unit * amount)

        row = {
            "h4_bias": state.h4_bias,
            "signal_time": state.signal_time,
            "entry_fill_time": entry_fill_time,
            "entry_delay_min": entry_delay_min,
            "side": state.side,
            "planned_entry_price": state.planned_entry_price,
            "entry_price": entry_price,
            "stop_loss": state.stop_loss,
            "take_profit": state.take_profit,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "result": "win" if pnl_per_unit > 0 else "loss",
            "r_multiple": r_multiple,
            "r_multiple_after_fee": r_after_fee,
            "fees_usdt": fees,
        }
        pd.DataFrame([row]).to_csv(
            LIVE_TRADES_CSV,
            mode="a",
            header=not os.path.exists(LIVE_TRADES_CSV),
            index=False,
            encoding="utf-8",
        )
        logger.info(
            f"已記錄交易: {state.side} | 進場 {entry_price:.2f} → 出場 {exit_price:.2f} | "
            f"{row['result']} | R={r_multiple if r_multiple is None else round(r_multiple, 2)}"
        )
        print_live_stats()
    except Exception as e:
        logger.warning(f"寫入交易紀錄失敗（不影響交易運作）: {e}")


def print_live_stats():
    """讀取實盤交易紀錄 CSV，印出累計統計（格式比照回測的績效報告）。"""
    if not os.path.exists(LIVE_TRADES_CSV):
        return
    try:
        df = pd.read_csv(LIVE_TRADES_CSV)
        df = df.dropna(subset=["r_multiple_after_fee"])
        if df.empty:
            return
        total = len(df)
        win_rate = (df["result"] == "win").mean()
        avg_r = df["r_multiple_after_fee"].mean()
        cum_r = df["r_multiple_after_fee"].cumsum()
        max_dd = (cum_r - cum_r.cummax()).min()
        logger.info(
            f"實盤累計統計 | 筆數 {total} | 勝率 {win_rate:.1%} | 平均R(已扣實際手續費) {avg_r:.3f} | "
            f"累積R {cum_r.iloc[-1]:.2f} | 最大回落 {max_dd:.2f}"
        )
    except Exception as e:
        logger.warning(f"讀取實盤統計失敗: {e}")


# ---------------------------------------------------------
# 下單數量計算（合約：用保證金 * 槓桿 算出名目金額，再換算成幣數量）
# ---------------------------------------------------------
def calculate_contract_amount(exchange, symbol, margin_usdt, leverage, price):
    notional = margin_usdt * leverage          # 名目曝險金額
    amount = notional / price
    amount = exchange.amount_to_precision(symbol, amount)
    return float(amount)


# ---------------------------------------------------------
# 下單：進場 + 停損 + 停利（三張單）
# ---------------------------------------------------------
def place_entry_with_sl_tp(exchange, symbol, plan, max_retries=3):
    """
    plan 來自 snr_strategy.find_multi_timeframe_setup() 的回傳，
    包含 signal / entry_price / stop_loss / take_profit
    """
    side = "buy" if plan["signal"] == "long" else "sell"
    close_side = "sell" if side == "buy" else "buy"   # 停損停利是反向單，用來平倉

    amount = calculate_contract_amount(
        exchange, symbol, MARGIN_USDT_PER_TRADE, LEVERAGE, plan["entry_price"]
    )

    for attempt in range(1, max_retries + 1):
        try:
            entry_order = exchange.create_order(symbol, "MARKET", side, amount)
            logger.info(f"進場成功: {side} {amount} {symbol} | order id={entry_order['id']}")
            break
        except ccxt.InsufficientFunds as e:
            logger.error(f"保證金不足，無法進場: {e}")
            return None
        except ccxt.NetworkError as e:
            logger.warning(f"網路錯誤，第 {attempt} 次重試進場: {e}")
            time.sleep(2 ** attempt)
        except ccxt.ExchangeError as e:
            logger.error(f"交易所回傳錯誤，不重試: {e}")
            return None
    else:
        logger.error("進場重試次數用盡，放棄這次交易")
        return None

    # 停損單：STOP_MARKET，reduceOnly 確保只會平倉、不會反向開新倉
    # 這是進場後第一優先要掛好的保護，如果掛不上去，寧可直接市價平倉，
    # 也不要讓一個完全沒有停損保護的部位留在市場上
    sl_order = None
    sl_params = {"stopPrice": exchange.price_to_precision(symbol, plan["stop_loss"]), "reduceOnly": True}
    try:
        sl_order = exchange.create_order(symbol, "STOP_MARKET", close_side, amount, None, sl_params)
        logger.info(f"停損單掛出: {sl_order['id']} @ {plan['stop_loss']:.2f}")
    except ccxt.BaseError as e:
        logger.error(f"停損單掛單失敗，部位目前完全沒有保護，立即市價平倉: {e}")
        try:
            exchange.create_order(symbol, "MARKET", close_side, amount, None, {"reduceOnly": True})
            logger.error("已緊急市價平倉。這筆交易異常結束，建議人工確認帳戶狀態。")
        except ccxt.BaseError as close_error:
            logger.critical(f"緊急平倉也失敗了！部位目前沒有任何保護，請立刻人工介入檢查帳戶：{close_error}")
        return None  # 不回傳部位資訊，因為部位應該已經（或嘗試）被平掉了

    # 停利單：TAKE_PROFIT_MARKET，同樣 reduceOnly
    tp_order = None
    tp_params = {"stopPrice": exchange.price_to_precision(symbol, plan["take_profit"]), "reduceOnly": True}
    try:
        tp_order = exchange.create_order(symbol, "TAKE_PROFIT_MARKET", close_side, amount, None, tp_params)
        logger.info(f"停利單掛出: {tp_order['id']} @ {plan['take_profit']:.2f}")
    except ccxt.BaseError as e:
        # 常見原因（例如 code -2021 "Order would immediately trigger"）：
        # 掛單當下，市價已經達到或超過停利目標價位了。這種情況下目標其實已經
        # 達到，與其讓停利單掛不上去、部位卻只剩停損保護，不如直接市價平倉，
        # 等同於手動在目標價達成的當下出場。
        logger.warning(f"停利單掛單失敗（常見原因是價格已達到停利目標）: {e}")
        logger.warning("改為直接市價平倉，視為已在目標價附近出場")
        try:
            # 停損單是 Algo Order，要用算法訂單專屬的撤銷端點（見 cancel_pending_sl_tp 的說明）
            exchange.fapiPrivateDeleteAlgoOrder({"symbol": exchange.market_id(symbol), "algoId": sl_order["id"]})
            logger.info(f"停損單 {sl_order['id']} 已撤銷")
        except ccxt.OrderNotFound:
            logger.info(f"停損單 {sl_order['id']} 已經不存在，略過")
        except ccxt.BaseError as cancel_error:
            # 一定要印出來：代表停損單可能還留著，請去交易所網頁手動確認
            logger.warning(f"停損單 {sl_order['id']} 撤銷失敗，請手動到交易所確認並撤銷：{cancel_error}")
        try:
            exchange.create_order(symbol, "MARKET", close_side, amount, None, {"reduceOnly": True})
            logger.info("已市價平倉")
            return None  # 部位已經平倉完畢，不需要外層繼續追蹤
        except ccxt.BaseError as close_error:
            # 連平倉都失敗，退而求其次：至少讓外層知道部位還在、還有停損保護，
            # 沒有停利單，下一輪迴圈會持續監控，不會憑空消失在程式的追蹤之外
            logger.critical(f"市價平倉失敗，部位目前只剩停損保護、沒有停利，請人工確認！錯誤: {close_error}")
            return entry_order, sl_order, None

    return entry_order, sl_order, tp_order


def cancel_pending_sl_tp(exchange, symbol, state: PositionState):
    """
    其中一邊（停損或停利）成交後，另一邊要手動取消，不然會留著一張廢單。

    STOP_MARKET / TAKE_PROFIT_MARKET 這類條件單，幣安已經遷移到獨立的
    「Algo Order（算法訂單）」服務，撤單要用專屬的 DELETE /fapi/v1/algoOrder
    端點、用 algoId 撤銷，不是一般訂單撤銷用的 orderId/endpoint。
    用一般的 exchange.cancel_order() 打這種訂單，可能完全沒打中正確端點，
    卻又不會報錯——這正是先前遇到「撤單沒報錯、但交易所上訂單還在」的原因。

    這裡改用 fapiPrivateDeleteAlgoOrder 這個隱式 API（根據下單時觀察到的
    fapiPrivatePostAlgoOrder 命名方式推論而來，还沒有在你的 ccxt 版本上
    實際驗證過，如果這個方法名稱在你的版本行不通，請把完整錯誤訊息貼回來，
    我們再查正確名稱）。

    不管哪種情況，都把結果明確記錄下來，不要再讓撤單失敗悄悄發生而不留痕跡。
    """
    market_id = exchange.market_id(symbol)
    for label, order_id in [("停損", state.sl_order_id), ("停利", state.tp_order_id)]:
        if order_id is None:
            continue
        try:
            exchange.fapiPrivateDeleteAlgoOrder({"symbol": market_id, "algoId": order_id})
            logger.info(f"{label}單 {order_id} 已撤銷")
        except ccxt.OrderNotFound:
            logger.info(f"{label}單 {order_id} 已經不存在（可能已成交或先前已撤銷），略過")
        except ccxt.BaseError as e:
            # 這個警告一定要注意：代表撤單可能沒有成功，請直接去交易所網頁
            # 手動確認、手動撤銷，不要假設程式會自己再試一次
            logger.warning(f"{label}單 {order_id} 撤銷失敗，請手動到交易所確認並撤銷：{e}")


# ---------------------------------------------------------
# 主邏輯
# ---------------------------------------------------------
def run_once(exchange, state: PositionState):
    if state.in_position:
        # 已有部位時，檢查停損或停利是否已經有一邊成交
        positions = exchange.fetch_positions([SYMBOL])
        current_position = next((p for p in positions if float(p.get("contracts", 0)) != 0), None)

        if current_position is None:
            # 部位已經不存在，代表停損或停利其中一張已經觸發成交
            logger.info("偵測到部位已平倉（停損或停利觸發），清理剩餘掛單")
            # 先記錄這筆交易的實際結果（用實際成交紀錄，不受 Algo Order 查詢影響）
            record_closed_trade(exchange, SYMBOL, state)
            cancel_pending_sl_tp(exchange, SYMBOL, state)
            # 用「現在」當作下次掃描的門檻，排除持倉期間曾經出現、
            # 但機器人當時沒去理會的舊訊號，對齊 backtest 用出場時間當門檻的邏輯
            exit_floor_time = pd.Timestamp.now(tz="UTC").replace(tzinfo=None)
            state.exit()
            state.scan_floor_time = exit_floor_time
        else:
            logger.info(f"目前持有部位: {state.side}，繼續等待停損/停利觸發")
        return

    # 沒有部位時，才去找新的進場訊號
    # limit 用 snr_strategy.LIVE_H4_LOOKBACK / LIVE_M5_LOOKBACK，跟回測
    # 模擬機器人輪詢時用的視窗大小保持一致（單一來源，見該檔案的註解）
    df_4h = fetch_ohlcv_df(exchange, SYMBOL, TIMEFRAME_H4, limit=LIVE_H4_LOOKBACK)
    df_5m = fetch_ohlcv_df(exchange, SYMBOL, TIMEFRAME_M5, limit=LIVE_M5_LOOKBACK)

    # 傳入 scan_floor_time，避免同一個已經交易過、或持倉期間被忽略的舊訊號
    # 在下一輪迴圈又被判定成「最早的訊號」而重複進場
    plan = find_multi_timeframe_setup(df_4h, df_5m, min_5m_time=state.scan_floor_time)
    if plan is None:
        logger.info("目前沒有符合條件的 SNR 訊號")
        return

    logger.info(
        f"找到訊號: {plan['signal']} | 4H偏向: {plan['bias_from_4h']} | "
        f"進場 {plan['entry_price']:.2f} | 停損 {plan['stop_loss']:.2f} | 停利 {plan['take_profit']:.2f}"
    )

    # 不管下單成功與否，都先記住這個訊號的進場時間，避免同一個訊號在下次
    # 迴圈又被重新判定一次（例如下單失敗，但訊號本身已經"處理過"了）
    state.scan_floor_time = plan["entry_time"]

    result = place_entry_with_sl_tp(exchange, SYMBOL, plan)
    if result is None:
        return

    entry_order, sl_order, tp_order = result
    # tp_order 可能是 None（停利單掛單失敗、連緊急平倉也失敗的極端情況）
    # 這時部位還是真實存在的，一定要記錄進 state，不然下一輪機器人會誤判成
    # 空手、跑去開第二筆部位，讓這筆部位變成沒人管理的孤兒倉位
    tp_order_id = tp_order["id"] if tp_order is not None else None
    state.enter(plan["signal"], entry_order["id"], sl_order["id"], tp_order_id)
    state.set_trade_details(
        planned_entry_price=plan["entry_price"],
        stop_loss=plan["stop_loss"],
        take_profit=plan["take_profit"],
        signal_time=plan["entry_time"],
        h4_bias=plan["bias_from_4h"],
    )
    if tp_order_id is None:
        logger.warning("注意：這筆部位目前沒有停利單，只有停損保護，請留意後續迴圈的狀態")


def recover_state_from_exchange(exchange, symbol):
    """
    程式啟動（包含重啟）時，先去交易所查詢目前真實的部位狀態，重建 state，
    而不是每次都假設自己是空手。沒有這一步，程式只要重啟一次，就會把
    交易所上實際還開著的部位當成不存在，跑去開第二筆重複的倉位。
    """
    state = PositionState()

    positions = exchange.fetch_positions([symbol])
    current_position = next((p for p in positions if float(p.get("contracts", 0)) != 0), None)

    if current_position is None:
        # 即使目前空手，也要設一個保守的掃描門檻，用「現在」當起點。
        # 不這樣做的話，scan_floor_time 會是 None，等於完全忘記「哪些訊號
        # 已經交易過」——如果重啟前曾經交易過某個訊號、後來平倉了，重啟後
        # 只要那個訊號還留在最近的資料視窗裡，會被當成全新訊號重新交易
        # 一次。這個問題已經實測發生過：重啟後重複交易了同一組進場/停損/
        # 停利價位。代價是：重啟前一瞬間如果剛好有一個「真正全新、還沒來
        # 得及交易」的訊號，也會被這個保守門檻一併排除，但這比重複交易
        # 已經結束的舊訊號安全得多。
        state.scan_floor_time = pd.Timestamp.now(tz="UTC").replace(tzinfo=None)
        logger.info("啟動檢查：交易所目前沒有未平倉部位，正常啟動")
        return state

    side = "long" if current_position.get("side") == "long" else "short"

    sl_order_id = None
    tp_order_id = None
    sl_price = None
    tp_price = None

    # STOP_MARKET / TAKE_PROFIT_MARKET 這類條件單，幣安已經遷移到獨立的
    # Algo Order 服務，fetch_open_orders() 查不到，要用 Algo Order 專屬的
    # 查詢端點 GET /fapi/v1/openAlgoOrders（這就是先前重啟後顯示「停損單:
    # 找不到、停利單: 找不到」的真正原因——不是訂單真的不見了，是查詢的
    # 地方查錯了）。這個隱式API名稱同樣是根據 fapiPrivatePostAlgoOrder /
    # fapiPrivateDeleteAlgoOrder 的命名方式推論而來，還沒有實際驗證過，
    # 如果呼叫失敗，把完整錯誤訊息貼回來，我們再查正確名稱。
    try:
        algo_orders = exchange.fapiPrivateGetOpenAlgoOrders({"symbol": exchange.market_id(symbol)})
        for order in algo_orders:
            order_type = str(order.get("orderType", "")).upper()
            algo_id = order.get("algoId")
            trigger = order.get("triggerPrice")
            trigger = float(trigger) if trigger not in (None, "") else None
            if "STOP" in order_type:
                sl_order_id = algo_id
                sl_price = trigger
            elif "PROFIT" in order_type:
                tp_order_id = algo_id
                tp_price = trigger
    except ccxt.BaseError as e:
        logger.warning(f"查詢 Algo Order 失敗，無法確認停損停利單是否存在：{e}")

    state.enter(side, entry_order_id=None, sl_order_id=sl_order_id, tp_order_id=tp_order_id)
    # 重啟後原本的訊號資訊（訊號時間、4H偏向）已經遺失，但停損/停利價位可以
    # 從交易所上的條件單還原，平倉時才算得出R倍數
    state.set_trade_details(
        planned_entry_price=float(current_position["entryPrice"]) if current_position.get("entryPrice") else None,
        stop_loss=sl_price,
        take_profit=tp_price,
        signal_time=None,
        h4_bias=None,
    )
    # 不知道這筆部位原本是什麼時候、用哪個5m訊號進場的，保守起見用「現在」當門檻，
    # 避免重啟後又去用一個時間對不起來的舊訊號
    state.scan_floor_time = pd.Timestamp.now(tz="UTC").replace(tzinfo=None)

    logger.warning(
        f"啟動檢查：偵測到重啟前已存在的未平倉部位 [{side}]，"
        f"停損單: {sl_order_id or '找不到'}，停利單: {tp_order_id or '找不到'}。"
        f"已重建狀態，不會重複進場。"
    )
    if sl_order_id is None:
        logger.critical("警告：這筆重啟後接手的部位找不到對應的停損單，目前可能完全沒有保護，請立即人工確認！")

    return state


def sleep_until_next_candle(period_seconds=CHECK_INTERVAL_SECONDS, offset_seconds=5):
    """
    睡到下一根5分鐘K線收盤後幾秒再醒來。
    原本的 time.sleep(300) 會因為每輪執行時間而慢慢漂移，輪詢時間點跟K線收盤
    的相對位置不固定，有時候訊號出現後最多要等將近5分鐘才會被發現、進場價也
    跟著偏離。對齊K線邊界後，機器人會在每根K線收盤後立刻判斷，跟回測假設的
    「在K線收盤時進場」一致。offset 是留給交易所把K線結算完成的緩衝。
    """
    now = time.time()
    next_boundary = (now // period_seconds + 1) * period_seconds
    time.sleep(max(0.0, next_boundary + offset_seconds - now))


def main():
    exchange = build_exchange()
    state = recover_state_from_exchange(exchange, SYMBOL)
    print_live_stats()  # 啟動時先顯示目前為止的實盤累計統計（如果有紀錄的話）

    logger.info(f"啟動 SNR 合約交易迴圈: {SYMBOL} | 每 {CHECK_INTERVAL_SECONDS} 秒檢查一次")

    while True:
        try:
            run_once(exchange, state)
        except Exception as e:
            logger.exception(f"主迴圈發生未預期錯誤: {e}")
        sleep_until_next_candle()


if __name__ == "__main__":
    main()