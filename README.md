# Expert Intelligence Trading Bot

A high-performance, real-time trading automation system for Deriv markets, featuring advanced technical analysis, fractal forecasting, Monte Carlo simulations, and robust risk management.

## 🚀 Core Features

- **9 Advanced Strategies**: From conservative daily bias crossovers to cutting-edge Monte Carlo statistical modeling.
- **Echo Forecast Engine**: Employs fractal similarity analysis to project future price paths based on historical patterns. Now integrated as a structural filter for intelligence strategies.
- **Monte Carlo Future Move Indicator**: Runs hundreds of simulations to provide probability distributions (Bull/Bear %) and optimized price paths.
- **Expiry Range Optimization**: For intelligence strategies (5, 6, 9), the bot scans a window of 1–10 ticks to find the optimal expiry where Echo confidence and MC probability are simultaneously highest.
- **Intelligent Exit Engine**:
    - **Early Exit Triggers**: Sells contracts early if price hits MC probability averages or if the Echo path flips direction mid-trade.
    - **Multiplier Management**: Aggressively trails SL in "Free Ride" mode once significant profit is reached.
- **Smart Target Engine**: Automatically sets Take Profit (TP) and Stop Loss (SL) for Multipliers using a combination of ATR volatility, Monte Carlo deviation bands, and Echo structural extremes.
- **Dynamic UI**: Responsive dashboard with real-time WebSocket updates, strategy-specific screener views, and integrated risk controls.

## 🏗 Architecture

- **`app.py`**: Flask & SocketIO server managing the dashboard and configuration API.
- **`bot_engine.py`**: The core execution hub. Processes ticks, manages account state, and orchestrates trades.
- **`handlers/screener_handler.py`**: Background analyzer providing real-time multi-timeframe scoring and forecasting.
- **`handlers/strategy_handler.py`**: Evaluates entry/exit conditions and enforces strict strategy rules.
- **`handlers/ta_handler.py`**: singleton manager for technical indicators and historical data.
- **`handlers/utils.py`**: Proprietary logic library (LuxAlgo SNR, Monte Carlo, Echo Forecast, Price Action).

## 📈 Strategy Rules & Logic

### 1. Slow (Daily/15m)
- **Entry**: Price crosses the Daily Open. Confirmed by 15m TA signals.
- **Expiry**: End of Day (EOD) or dynamic move > 2 Daily ATRs.
- **Display**: Trend indicators and Daily Bias status.

### 2. Moderate (1h/3m)
- **Entry**: Price crosses the 1h Open. Confirmed by 3m TA signals.
- **Expiry**: End of Hour (EOH).

### 3. Fast (15m/1m)
- **Entry**: Price crosses the 15m Open. Confirmed by 1m TA signals.
- **Expiry**: End of 15m Period.

### 4. SNR Breakout Reversal (LuxAlgo)
- **Logic**: Uses strictly defined LuxAlgo 15/15 pivot points to find Support (S) and Resistance (R) zones.
- **Entry**: Triggers on a reversal breakout from a tested zone confirmed by 1m PA patterns (Pin Bar, Engulfing).
- **Exclusive**: The only strategy allowed to display SNR zones on the dashboard.

### 5. Synthetic Intelligence
- **Scoring**: Weighted aggregate of Trend (EMA 50/200), Momentum (RSI/Stoch), and Volatility (ATR).
- **Structure**: Uses Echo Forecast direction as the primary price action filter.
- **Optimization**: Employs Monte Carlo Expiry Range Optimization (1-10 ticks).

### 6. Intelligence Legacy
- **Logic**: Classic indicator-heavy approach (RSI, Bollinger Bands, MACD) refined with modern filters.
- **Structure**: Echo Forecast path agreement required.
- **Optimization**: Monte Carlo inflection point detection for smart expiry.

### 7. Multi-TF Alignment
- **Entry**: Strictly based on triple TA-filter alignment across Small, Mid, and High timeframes. (No Echo/RR gates).
- **Expiry**: Dynamic based on signal strength:
    - High TF Strong: 20m | High TF Standard: 60m
    - Mid TF Strong: 1-4m range (confidence based)
    - Aligned Mid/Small: 20m
    - Default: 5m

### 8. UT Bot Alerts
- **Logic**: ATR-based Trailing Stop alerts (PineScript v4 port).
- **Entry**: 1m UT Buy/Sell signals confirmed by secondary TA filters.
- **Multipliers**: Uses highest available multiplier range for aggressive growth.

### 9. Echo + Monte Carlo Evolution
- **Entry**: Fired when Echo projected path, Monte Carlo bias (>55%), and 1m TA signals all agree.
- **Optimization**: Scans 1-10 steps to find the peak of (Echo Score × MC Probability).
- **Exits**: MC average line hit or Echo path flip.

## 🛠 Setup

1. **Install**: `pip install -r requirements.txt` (Note: `talib` is not required; uses native `ta` library).
2. **Configure**: Enter Deriv API Token in dashboard settings.
3. **Run**: `python app.py`

## 🛡 Risk Management
- **TP/SL**: Strictly visible and calculated for **Multiplier** mode only.
- **Daily Limits**: Bot automatically pauses trading if the Daily Loss % or Profit % threshold is hit.
- **Adaptive Sensitivity**: Increases entry thresholds automatically after 3 consecutive losses on a symbol.

---
*Built for advanced algorithmic trading on Deriv.*
