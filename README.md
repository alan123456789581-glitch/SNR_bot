# SNR Crypto Strategy: 回測與實驗紀錄

用 Python + ccxt 研究「支撐/壓力(SNR)」策略在幣安 USDT 本位合約(BTC 永續)上的表現。
這個專案的重點不是找到一個賺錢的策略,而是**用可重現、事先定標準的流程檢驗想法,並誠實報告被否定的結果**。

## 策略規則

1. **4H 判斷方向:** 取相鄰兩根 K 線的實體(open/close)作為支撐/壓力區;區域被測試後判斷 long 或 short
2. **5M 進場:** 切到 5 分鐘,尋找與 4H 同方向的 regular 或 breakout(flipped SNR)型態,出現後進場
3. **風險報酬:** 回測使用 1:2 盈虧比;停損約 0.4%

## 主要結果

| 版本 | 筆數 | 勝率 | 毛均 R | 淨均 R |
|---|---|---|---|---|
| 基準 | 1763 | 31.7% | −0.049 | −0.247 |
| 假設二:4H 方向 24 小時內有效 | 650 | 33.5% | +0.006 | −0.192 |

假設一(位置限制)同樣沒有改善,數據見 [`experiments.md`](experiments.md)。

**結論:** 手續費每筆約 0.2R,而毛期望值約為 0,因此扣費後為負。這組規則在 BTC 永續合約 5 分鐘進場上,沒有扣完手續費仍能獲利的優勢。
詳細的假設、成功標準與每次實驗的數據見 [`experiments.md`](experiments.md)。

## 研究方法

- **防止偷看未來:** 檢查並移除 look-ahead bias,訊號一律只用已收盤的 K 線
- **事先定義成功標準:** 每個假設在跑回測前先寫下通過條件(例如樣本數 ≥ 300、毛期望值的信賴區間、樣本內外一致)
- **多重檢定自律:** 兩個假設都被否定後,不再繼續加假設調參
- **不做事後反向操作:** 顯著為負的結果不直接反著做,避免資料窺探,且仍會被手續費吃掉

## 專案結構

```
├── README.md
├── experiments.md                # 各假設、成功標準與實驗結果
├── snr_strategy.py               # 策略邏輯(4H 方向 + 5M 進場訊號)
├── backtest_snr.py               # 回測
├── backtest_sensitivity.py       # 參數敏感度分析
├── compare_live_vs_backtest.py   # Demo 實盤紀錄與回測的比對
├── snr_bot.py                    # ccxt 交易機器人(Demo Trading)
├── test_expiry.py                # 4H 方向有效期限的測試
└── .gitignore
```

## 安裝與執行

```bash
pip install -r requirements.txt
```

機器人從環境變數讀取金鑰,請在專案根目錄建立 `.env`(不會被上傳):

```
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
```

```bash
python backtest_snr.py     # 跑回測
python snr_bot.py          # 啟動機器人(僅限 Demo Trading)
```

## 注意事項

- 本專案僅供研究與學習,**不構成投資建議**
- 機器人只在幣安 Demo Trading 測試,目前建議停用
- 請勿將真實 API 金鑰提交到版本控制

## 後續方向

換一個全新的研究問題,而不是繼續調整這組規則。