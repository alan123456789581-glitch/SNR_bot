"""
test_expiry.py：檢查假設二的方向有效期限——4H 方向確立超過期限後，5M 訊號不再進場。
用 test_proximity.py 同樣的手工K線：4H 有一個壓力區間並產生 short 方向，
5M 的 short 訊號分別出現在方向確立後約 2 小時（有效）、約 30 小時（過期）。
執行：py tests/test_expiry.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd
import snr_strategy as S

def frame(rows, start, freq):
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)

base4 = [(100, 100.6, 99.6, 100.2)] * 6
rows4 = base4 + [(100.2, 101.5, 100.1, 101.0), (101.0, 101.2, 100.0, 100.4),
                 (100.4, 100.5, 99.7, 99.9), (99.9, 100.2, 99.6, 100.0), (100.0, 101.4, 99.9, 100.3)]
df4 = frame(rows4, "2026-01-01 00:00", "4h")
h4_close = df4.index[-1] + S.H4_PERIOD

def df5_after(hours, L=101.3):
    """方向確立後 hours 小時才開始出現的 5M short 訊號（最後一根K線 = 訊號K線，進場價 L - 0.5）"""
    b = [(L - 0.5, L - 0.2, L - 0.8, L - 0.5)] * 25
    rows = b + [(L - 0.5, L + 0.6, L - 0.55, L), (L, L + 0.1, L - 0.6, L - 0.4),
                (L - 0.4, L - 0.3, L - 0.9, L - 0.8), (L - 0.8, L + 0.2, L - 0.85, L - 0.5)]
    return frame(rows, h4_close + pd.Timedelta(hours=hours), "5min")

ok_all = True
def check(name, got, want):
    global ok_all
    good = (got == want); ok_all &= good
    print(f"{'✅' if good else '❌'} {name}: 實際 {got} | 預期 {want}")

def entered(plan): return plan is not None

S.H4_ZONE_PROXIMITY_PCT = None          # 只測有效期限，位置限制關閉
S.H4_BIAS_EXPIRY_HOURS = 24
fresh, stale = df5_after(2), df5_after(30)
check("方向確立後約 2 小時 + 期限 24 小時 → 進場", entered(S.find_multi_timeframe_setup(df4, fresh, lookback=5)), True)
check("方向確立後約 30 小時 + 期限 24 小時 → 不進場", entered(S.find_multi_timeframe_setup(df4, stale, lookback=5)), False)
check("方向確立後約 30 小時 + 明確關閉(inf) → 進場", entered(S.find_multi_timeframe_setup(df4, stale, lookback=5, h4_bias_expiry_hours=float("inf"))), True)
check("方向確立後約 2 小時 + 期限 1 小時 → 不進場", entered(S.find_multi_timeframe_setup(df4, fresh, lookback=5, h4_bias_expiry_hours=1)), False)
S.H4_BIAS_EXPIRY_HOURS = None
check("全域設成 None（關閉）→ 過期也進場", entered(S.find_multi_timeframe_setup(df4, stale, lookback=5)), True)
S.H4_BIAS_EXPIRY_HOURS = 24
p = S.find_multi_timeframe_setup(df4, fresh, lookback=5)
check("plan 記錄方向已確立的時數（約 4.4 小時）", p is not None and 4.0 < p["h4_bias_age_hours"] < 5.0, True)
print("\n全部通過" if ok_all else "\n有失敗")