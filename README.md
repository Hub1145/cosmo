# Expert Intelligence Trading Bot

A high-performance, real-time trading automation system for Deriv markets, featuring advanced technical analysis, fractal forecasting, and robust risk management.

## 🚀 Core Features

- **8 Multi-Timeframe Strategies**: From conservative trend following (Slow) to aggressive scalping (UT Bot Alerts).
- **Echo Forecast Engine**: Employs fractal similarity analysis to project future price paths based on historical patterns.
- **Expert Intelligence Expiry**: Dynamically calculates the optimal trade duration using "Profitable Arrival" and alignment logic.
- **Smart Target Engine**: Automatically sets Take Profit (TP) and Stop Loss (SL) based on ATR volatility and market structure.
- **Real-Time Screener**: A dynamic dashboard providing a bird's-eye view of all symbols with multi-strategy scoring and confidence metrics.
- **Robust Risk Management**:
    - Per-position TP/SL management.
    - Real-time realized + floating PnL tracking.
    - Daily profit targets and loss limits with automatic trading pause.
- **Dynamic UI**: Modern, responsive dashboard with Light/Dark mode, real-time WebSocket updates, and loading feedback.

## 🏗 Architecture

The project is built with a decoupled architecture for stability and performance:

- **`app.py`**: The entry point. A Flask & SocketIO server that hosts the web dashboard and manages configuration APIs.
- **`bot_engine.py`**: The core execution engine. Handles the main Deriv WebSocket connection, tick-by-tick data processing, order execution, and account state management.
- **`handlers/ta_handler.py`**: Manages technical indicator computations and maintains a dedicated connection for historical data retrieval. Uses a singleton `ConnectionManager` for efficiency.
- **`handlers/screener_handler.py`**: The "brains" behind the UI. It runs a dedicated background loop to analyze multiple symbols across various timeframes, providing the metadata used for signals and dashboard updates.
- **`handlers/strategy_handler.py`**: The trade orchestrator. Evaluates entry/exit logic based on active strategies and enforces risk management gates.
- **`handlers/utils.py`**: A specialized library containing proprietary trading logic:
    - **LuxAlgo Style SNR**: Support and Resistance zone detection using pivot logic.
    - **Structural RR**: Calculates Reward-to-Risk ratios based on structural forecast extremes.
    - **Price Action Patterns**: Automatic detection of Pin Bars, Engulfing, and Marubozu candles.

## 📈 Strategies

1.  **Slow (Daily/15m)**: Trend-following with a daily bias.
2.  **Moderate (1h/3m)**: Intermediate timeframe crossover logic.
3.  **Fast (15m/1m)**: High-frequency scalp targeting quick moves.
4.  **SNR Breakout Reversal**: Strategic entries at LuxAlgo-style Support & Resistance zones.
5.  **Synthetic Intelligence**: A high-confidence aggregator using multiple indicators and alignment.
6.  **Intelligence Legacy**: Classic scalping methodology refined with ATR-based filtering.
7.  **Multi-TF Alignment**: A rigorous triple-confirmation filter (available in Multi-TF mode).
8.  **UT Bot Alerts**: Optimized implementation of the popular UT Bot Trailing Stop alerts.

## 🛠 Setup & Installation

1.  **Prerequisites**: Python 3.10+, Deriv Account.
2.  **Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configuration**:
    - Launch the dashboard and enter your Deriv API Token and App ID in the Settings modal.
    - Configure your risk limits (Daily Max Loss/Profit) and preferred active strategy.
4.  **Running the Bot**:
    ```bash
    python app.py
    ```
    Access the dashboard at `http://localhost:5000`.

## 🛡 Risk Disclaimer

Trading financial instruments involves significant risk and can result in the loss of your invested capital. This bot is a tool for automation and does not guarantee profits. Always test in a Demo account first and use responsible risk management settings.

---
*Built with ❤️ for advanced traders.*
