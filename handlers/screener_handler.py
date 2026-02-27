import time
import logging
import asyncio
import threading
import pandas as pd
import ta
from concurrent.futures import ThreadPoolExecutor
from handlers.ta_handler import get_ta_signal, get_ta_indicators, fetch_candles, manager
from handlers.utils import (
    calculate_snr_zones, check_price_action_patterns, score_reversal_pattern,
    predict_expiry, calculate_echo_forecast, calculate_structural_rr, get_smart_targets,
    calculate_supertrend, calculate_fractals, calculate_order_blocks, detect_macd_divergence
)

class ScreenerHandler:
    def __init__(self, bot_engine):
        self.bot = bot_engine
        self.stop_event = bot_engine.stop_event

    def update_screener(self, symbol, config):
        try:
            strat_key = config.get('active_strategy', 'strategy_1')
            if strat_key == 'strategy_5':
                return self.analyze_strategy_5(symbol)
            elif strat_key == 'strategy_6':
                return self.analyze_strategy_6(symbol)
            elif strat_key == 'strategy_7':
                return self.update_strat7_analysis(symbol, config)
            elif strat_key == 'strategy_1':
                return self.analyze_crossover_strategy(symbol, 1, "15m", 86400)
            elif strat_key == 'strategy_2':
                return self.analyze_crossover_strategy(symbol, 2, "3m", 3600)
            elif strat_key == 'strategy_3':
                return self.analyze_crossover_strategy(symbol, 3, "1m", 900)
            elif strat_key == 'strategy_4':
                return self.analyze_strategy_4(symbol)
            elif strat_key == 'strategy_8':
                return self.analyze_strategy_8(symbol)
            return None
        except Exception as e:
            logging.error(f"Screener error for {symbol}: {e}")
            return None

    def _get_htf_countdown(self, granularity_sec):
        """Calculates seconds remaining until the next candle boundary."""
        now = time.time()
        if granularity_sec <= 0: return 60
        next_boundary = ((int(now) // granularity_sec) + 1) * granularity_sec
        return int(next_boundary - now)

    def _get_smart_expiry(self, df_ltf, ltf_min, htf_min):
        try:
            if df_ltf is None or len(df_ltf) < 20:
                return htf_min
            atr_series = ta.volatility.AverageTrueRange(df_ltf['high'], df_ltf['low'], df_ltf['close'], window=14).average_true_range()
            atr_current = atr_series.iloc[-1]
            atr_avg = atr_series.rolling(50).mean().iloc[-1]
            if not atr_current or not atr_avg:
                return htf_min
            ratio = atr_avg / atr_current
            if ltf_min == 1:
                suggested = 3 * ratio
                return max(1, min(5, int(round(suggested))))
            elif ltf_min == 5:
                suggested = 12 * ratio
                return max(5, min(20, int(round(suggested))))
            else:
                mid = (ltf_min + htf_min) / 2
                suggested = mid * ratio
                return max(ltf_min, min(htf_min * 3, int(round(suggested))))
        except:
            return htf_min

    def _calculate_scores(self, symbol, indicators, df):
        try:
            if df is None or df.empty:
                return 5.0, 5.0, 5.0, 5.0

            # 1. Trend Score (EMA alignment)
            ema20 = indicators.get('ema20', df['close'].ewm(span=20).mean().iloc[-1])
            ema50 = indicators.get('ema50', df['close'].ewm(span=50).mean().iloc[-1])
            price = df['close'].iloc[-1]
            trend = 5.0
            if price > ema20 > ema50: trend = 8.5
            elif price < ema20 < ema50: trend = 1.5

            # 2. Momentum Score (RSI)
            rsi = indicators.get('rsi', 50)
            momentum = round(rsi / 10, 1)

            # 3. Volatility Score (ATR vs MA)
            atr_series = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            atr_curr = atr_series.iloc[-1]
            atr_avg = atr_series.rolling(50).mean().iloc[-1]
            volatility = round((atr_curr / atr_avg) * 5, 1) if atr_avg else 5.0

            # 4. Structure Score (Price Action)
            candles = []
            for _, row in df.tail(10).iterrows():
                candles.append({'open': row['open'], 'high': row['high'], 'low': row['low'], 'close': row['close']})
            pattern = check_price_action_patterns(candles)
            structure = 5.0
            if pattern:
                if "bullish" in pattern: structure = 7.5
                elif "bearish" in pattern: structure = 2.5

            return trend, momentum, volatility, structure
        except Exception as e:
            logging.error(f"Error calculating scores for {symbol}: {e}")
            return 0, 0, 0, 0

    def analyze_strategy_5(self, symbol):
        """Strategy 5: Synthetic Intelligence Screener"""
        try:
            # 1. Gather Data across multiple timeframes
            df1m = asyncio.run_coroutine_threadsafe(fetch_candles(symbol, "1m"), manager.loop).result()
            df5m = asyncio.run_coroutine_threadsafe(fetch_candles(symbol, "5m"), manager.loop).result()
            df1h = asyncio.run_coroutine_threadsafe(fetch_candles(symbol, "1h"), manager.loop).result()

            if df1m.empty or df5m.empty or df1h.empty: return None

            # Trend Block: EMA 50/200, SuperTrend, ADX
            ind1h = get_ta_indicators(symbol, "1h")
            ema50_1h = ind1h.get('ema50', 0)
            ema200_1h = ind1h.get('ema200', 0)
            adx_1h = ind1h.get('adx', 0)
            price_1h = ind1h.get('close', 0)

            st_val, st_dir = calculate_supertrend(df1h)
            st_curr = st_dir.iloc[-1] if not st_dir.empty else 0 # 1 for UP, -1 for DOWN

            trend_score = 0
            if ema50_1h and ema200_1h:
                if price_1h > ema50_1h > ema200_1h: trend_score += 30
                elif price_1h < ema50_1h < ema200_1h: trend_score -= 30

            if st_curr == 1: trend_score += 10
            else: trend_score -= 10

            if adx_1h > 25: trend_score *= 1.2 # Strength boost

            # Momentum Block: RSI, Stoch RSI, MACD Divergence
            ind5m = get_ta_indicators(symbol, "5m")
            rsi_5m = ind5m.get('rsi', 50)
            stoch_k_5m = ind5m.get('stoch_k', 50)
            macd_div = detect_macd_divergence(df5m)

            mom_score = 0
            if rsi_5m > 50: mom_score += 10
            else: mom_score -= 10

            if stoch_k_5m > 50: mom_score += 10
            else: mom_score -= 10

            if macd_div == 1: mom_score += 15
            elif macd_div == -1: mom_score -= 15

            # Volatility Block: ATR Relative
            atr_5m = ta.volatility.AverageTrueRange(df5m['high'], df5m['low'], df5m['close'], window=14).average_true_range()
            atr_curr = atr_5m.iloc[-1] if not atr_5m.empty else 0
            atr_avg_series = atr_5m.rolling(50).mean()
            atr_avg = atr_avg_series.iloc[-1] if not atr_avg_series.empty else 0
            vol_rel = (atr_curr / atr_avg) if atr_avg else 1.0

            # Map volatility to score (0.5x to 1.5x ATR map to -5 to +5)
            vol_score = (vol_rel - 1.0) * 10
            vol_score = max(-5, min(5, vol_score))

            # Structure Block: 5m Fractals (Scalp) or 1H Order Blocks (Multiplier)
            is_multiplier = self.bot.config.get('contract_type') == 'multiplier'
            struct_score = 0
            if is_multiplier:
                obs = calculate_order_blocks(df1h)
                for ob in obs:
                    if not ob['price']: continue
                    dist = abs(price_1h - ob['price'])/ob['price']
                    if dist < 0.01:
                        weight = 20 * (1 - dist/0.01)
                        if ob['type'] == 'Bullish OB': struct_score += weight
                        else: struct_score -= weight
            else:
                f_high, f_low = calculate_fractals(df5m)
                # Recent fractal lookback
                if any(f_low.tail(3)): struct_score += 15
                elif any(f_high.tail(3)): struct_score -= 15

            # Normalization and Confidence (Weights: Trend 40%, Momentum 30%, Vol 10%, Struct 20%)
            # We map scores to a 0.0 - 10.0 range for the UI
            norm_trend = max(0.0, min(10.0, 5.0 + (trend_score / 8.0)))
            norm_mom = max(0.0, min(10.0, 5.0 + (mom_score / 7.0)))
            norm_vol = max(0.0, min(10.0, 5.0 + (vol_score)))
            norm_struct = max(0.0, min(10.0, 5.0 + (struct_score / 4.0)))

            # Total Confidence Calculation
            total_raw = trend_score + mom_score + vol_score + struct_score
            confidence = min(100, abs(total_raw))
            direction = "CALL" if total_raw > 0 else "PUT"
            signal = "WAIT"

            # Adaptive Sensitivity
            sd = self.bot.symbol_data.get(symbol, {})
            loss_streak = sd.get('consecutive_losses', 0)

            threshold = 68 if is_multiplier else 72
            if loss_streak >= 3:
                threshold += (loss_streak - 2) * 5
                self.bot.log(f"Adaptive Sensitivity: Boosting Strategy 5 threshold to {threshold}% for {symbol} due to {loss_streak} losses.")

            if confidence >= threshold:
                signal = "BUY" if total_raw > 0 else "SELL"

            # 5. Echo Forecast (5m) validation
            fcast_prices, correlation = calculate_echo_forecast(df5m, projection='Outcome')
            if fcast_prices:
                fcast_data = {
                    'final': fcast_prices[-1], 'correlation': correlation,
                    'forecast_prices': fcast_prices,
                    'high': max(fcast_prices), 'low': min(fcast_prices),
                    'direction': "CALL" if fcast_prices[-1] > df5m['close'].iloc[-1] else "PUT"
                }
            else:
                fcast_data = {}

            atr_series = ta.volatility.AverageTrueRange(df5m['high'], df5m['low'], df5m['close']).average_true_range()
            atr_val = atr_series.iloc[-1] if not atr_series.empty else 0
            expiry, is_aligned = predict_expiry(symbol, 'strategy_5', 1, 60, confidence, fcast_data, df1m, direction=direction)

            # Alignment Filter
            if not is_aligned:
                signal = "WAIT"

            tp_price, sl_price, rr = None, None, 0
            # Always calculate targets for screener if possible
            tp_price, sl_price = get_smart_targets(df1m['close'].iloc[-1], 'long' if total_raw > 0 else 'short', atr_val, confidence, fcast_data)
            if fcast_prices:
                rr = calculate_structural_rr(df1m['close'].iloc[-1], fcast_prices, "BUY" if total_raw > 0 else "SELL", atr_val)

            # Relevance Logic for Strategy 5
            label = "Screener - Position Mode" if is_multiplier else "Screener - Signal Mode"
            price = df1m['close'].iloc[-1]
            # Always provide SNR count for Screener (1H Order Blocks)
            from handlers.utils import calculate_snr_zones
            snr_zones = calculate_snr_zones(symbol, {'htf_candles': df1h.to_dict('records')}, 3600)
            snr_count = len(snr_zones)

            if is_multiplier:
                obs = calculate_order_blocks(df1h)
                near_ob = None
                for ob in obs:
                    if not ob['price']: continue
                    if abs(price_1h - ob['price'])/ob['price'] < 0.01:
                        near_ob = ob
                        break
                if near_ob:
                    dist_ob = ((price_1h - near_ob['price']) / near_ob['price']) * 100
                    desc = f"NEAR {near_ob['type']} ({dist_ob:+.2f}%)"
                else:
                    desc = f"NO NEAR OB | CONF: {confidence}%"
            else:
                desc = f"CONF: {confidence}% | TREND: {norm_trend:.1f} | MOM: {norm_mom:.1f}"

            data = {
                'tp': round(tp_price, 4) if tp_price else None,
                'sl': round(sl_price, 4) if sl_price else None,
                'rr': round(float(rr), 1),
                'signal': signal,
                'direction': direction,
                'label': label,
                'desc': desc,
                'confidence': round(float(confidence), 1),
                'threshold': threshold,
                'expiry_min': expiry,
                'expiry_countdown': expiry * 60,
                'atr': round(atr_val, 4),
                'price': round(price_1h, 4),
                'snr_count': snr_count,
                'correlation': round(correlation, 2) if 'correlation' in locals() else 0,
                'trend': round(norm_trend, 1), 'momentum': round(norm_mom, 1),
                'volatility': round(norm_vol, 1), 'structure': round(norm_struct, 1),
                'fcast_data': fcast_data,
                'last_update': time.time()
            }
            self.bot.screener_data[symbol] = data
            self.bot.emit('screener_update', {'symbol': symbol, 'data': data})
            return data
        except Exception as e:
            logging.error(f"Error in Strategy 5 analysis for {symbol}: {e}", exc_info=True)
            return None

    def analyze_strategy_6(self, symbol):
        """Strategy 6: Intelligence Legacy"""
        try:
            # 1. Gather Data (1m, 5m, 1h, 4h)
            df1m = asyncio.run_coroutine_threadsafe(fetch_candles(symbol, "1m"), manager.loop).result()
            df5m = asyncio.run_coroutine_threadsafe(fetch_candles(symbol, "5m"), manager.loop).result()
            df1h = asyncio.run_coroutine_threadsafe(fetch_candles(symbol, "1h"), manager.loop).result()
            df4h = asyncio.run_coroutine_threadsafe(fetch_candles(symbol, "4h"), manager.loop).result()

            if df1m.empty or df5m.empty or df1h.empty or df4h.empty: return None

            # Indicator Blocks with Weights: Trend (3), Momentum (2), Volatility (1), Structure (2)
            ind1h = get_ta_indicators(symbol, "1h")
            ind1m = get_ta_indicators(symbol, "1m")
            ind4h = get_ta_indicators(symbol, "4h")

            # Trend Score (Weight 3)
            trend_score = 0
            c1h, e1h = ind1h.get('close'), ind1h.get('ema50')
            if c1h is not None and e1h is not None:
                if c1h > e1h: trend_score += 1
                else: trend_score -= 1

            c4h, e4h = ind4h.get('close'), ind4h.get('ema50')
            if c4h is not None and e4h is not None:
                if c4h > e4h: trend_score += 1
                else: trend_score -= 1

            c1m, e1m = ind1m.get('close'), ind1m.get('ema50')
            if c1m is not None and e1m is not None:
                if c1m > e1m: trend_score += 1
                else: trend_score -= 1
            trend_final = trend_score * 3

            # Momentum Score (Weight 2)
            mom_score = 0
            r1h = ind1h.get('rsi')
            if r1h is not None:
                if r1h > 50: mom_score += 1
                else: mom_score -= 1

            r1m = ind1m.get('rsi')
            if r1m is not None:
                if r1m > 50: mom_score += 1
                else: mom_score -= 1
            mom_final = mom_score * 2

            # Volatility Score (Weight 1) - Based on BB position
            bb_h = ind1m.get('bb_h')
            bb_l = ind1m.get('bb_l')
            price = ind1m.get('close', 0)
            vol_score = 0
            if bb_h is not None and bb_l is not None and price is not None:
                if bb_h > bb_l:
                    # 1.0 at upper band, -1.0 at lower band
                    vol_score = (price - (bb_h + bb_l)/2) / (bb_h - bb_l) * 2
            vol_final = max(-1.0, min(1.0, vol_score)) * 1

            # Structure Score (Weight 2)
            struct_score = 0
            macd_div = detect_macd_divergence(df1h)
            if macd_div == 1: struct_score += 1
            elif macd_div == -1: struct_score -= 1

            # Add SNR proximity to Strategy 6 structure
            # Use the df1h we just fetched to ensure SNR zones are calculated even if sd is empty
            from handlers.utils import calculate_snr_zones
            snr_zones = calculate_snr_zones(symbol, {'htf_candles': df1h.to_dict('records')}, 3600)
            if price:
                for z in snr_zones:
                    if abs(price - z['price'])/price < 0.002:
                        if z['type'] == 'S': struct_score += 0.5
                        elif z['type'] == 'R': struct_score -= 0.5

            struct_final = max(-1, min(1, struct_score)) * 2

            # Normalize confidence (max possible absolute score is 3*3 + 2*2 + 1*1 + 1*2 = 16)
            total_score = trend_final + mom_final + vol_final + struct_final
            confidence = min(100, abs(total_score) / 16 * 100)

            direction = "CALL" if total_score > 0 else "PUT"
            signal = "WAIT"
            if confidence >= 60:
                signal = "BUY" if total_score > 0 else "SELL"

            # Normalize scores for UI (0-10)
            norm_trend = max(0.0, min(10.0, 5.0 + (trend_score * 1.6)))
            norm_mom = max(0.0, min(10.0, 5.0 + (mom_score * 2.5)))
            norm_vol = max(0.0, min(10.0, 5.0 + (vol_score * 5.0)))
            norm_struct = max(0.0, min(10.0, 5.0 + (struct_score * 5.0)))

            # 5. Echo Forecast (5m) validation for Smart Expiry
            # We use 5m data instead of 1h for more precise scalp-horizon timing
            fcast_prices, correlation = calculate_echo_forecast(df5m, projection='Outcome')
            if fcast_prices:
                fcast_data = {
                    'final': fcast_prices[-1], 'correlation': correlation, 'forecast_prices': fcast_prices,
                    'high': max(fcast_prices), 'low': min(fcast_prices),
                    'direction': "CALL" if fcast_prices[-1] > df5m['close'].iloc[-1] else "PUT"
                }
            else:
                fcast_data = {}

            atr_series = ta.volatility.AverageTrueRange(df5m['high'], df5m['low'], df5m['close']).average_true_range()
            atr_val = atr_series.iloc[-1] if not atr_series.empty else 0
            # Use 60-minute window for Legacy Smart Expiry
            expiry, is_aligned = predict_expiry(symbol, 'strategy_6', 1, 60, confidence, fcast_data, df1m, direction=direction)

            # Alignment Filter
            if not is_aligned and signal != "WAIT":
                self.bot.log(f"Alignment Check: Suppressing Strategy 6 {signal} for {symbol} - Directional forecast mismatch.")
                signal = "WAIT"

            tp_price, sl_price, rr = None, None, 0
            tp_price, sl_price = get_smart_targets(df1m['close'].iloc[-1], 'long' if total_score > 0 else 'short', atr_val, confidence, fcast_data)
            if fcast_prices:
                rr = calculate_structural_rr(df1m['close'].iloc[-1], fcast_prices, "BUY" if total_score > 0 else "SELL", atr_val)

            # Relevance Logic for Strategy 6
            is_multiplier = self.bot.config.get('contract_type') == 'multiplier'
            label = "Legacy - Multiplier" if is_multiplier else "Legacy - Scalp"
            price_now = df1m['close'].iloc[-1]
            
            # Simple context for Legacy
            desc = f"CORR: {correlation:.2f}"

            data = {
                'tp': round(tp_price, 4) if tp_price else None,
                'sl': round(sl_price, 4) if sl_price else None,
                'rr': round(float(rr), 1),
                'signal': signal,
                'direction': direction,
                'label': label,
                'desc': desc,
                'confidence': round(float(confidence), 1),
                'threshold': 60,
                'expiry_min': expiry,
                'expiry_countdown': expiry * 60,
                'atr': round(atr_val, 4),
                'price': round(price_now, 4),
                'snr_count': len(snr_zones),
                'trend': round(norm_trend, 1), 'momentum': round(norm_mom, 1),
                'volatility': round(norm_vol, 1), 'structure': round(norm_struct, 1),
                'fcast_data': fcast_data,
                'last_update': time.time()
            }
            self.bot.screener_data[symbol] = data
            self.bot.emit('screener_update', {'symbol': symbol, 'data': data})
            return data
        except Exception as e:
            logging.error(f"Error in Strategy 6 analysis for {symbol}: {e}", exc_info=True)
            return None

    def update_strat7_analysis(self, symbol, config):
        tf_small_str = config.get('strat7_small_tf', '60')
        tf_mid_str = config.get('strat7_mid_tf', '300')
        tf_high_str = config.get('strat7_high_tf', '3600')

        def val_to_str(val):
            if val == 'OFF': return None
            val = int(val)
            if val == 60: return "1m"
            if val == 120: return "2m"
            if val == 180: return "3m"
            if val == 300: return "5m"
            if val == 600: return "10m"
            if val == 900: return "15m"
            if val == 1800: return "30m"
            if val == 3600: return "1h"
            if val == 86400: return "1d"
            return "1m"

        try:
            s_tf = val_to_str(tf_small_str)
            m_tf = val_to_str(tf_mid_str)
            h_tf = val_to_str(tf_high_str)

            price = 0

            rec_small = get_ta_signal(symbol, s_tf) if s_tf else "OFF"
            rec_mid = get_ta_signal(symbol, m_tf) if m_tf else "OFF"
            rec_high = get_ta_signal(symbol, h_tf) if h_tf else "OFF"

            active_recs = [r for r in [rec_small, rec_mid, rec_high] if r != "OFF"]

            label = "NEUTRAL"
            direction = "NEUTRAL"
            signal = "WAIT"

            if len(active_recs) == 1:
                rec = active_recs[0]
                if "BUY" in rec:
                    label = "ALIGNED_BUY"
                    direction = "CALL"
                    signal = "BUY"
                elif "SELL" in rec:
                    label = "ALIGNED_SELL"
                    direction = "PUT"
                    signal = "SELL"
            elif len(active_recs) > 1:
                all_buy = all("BUY" in r for r in active_recs)
                all_sell = all("SELL" in r for r in active_recs)

                # High TF check for QUICK signals
                high_rec = active_recs[-1]

                if all_buy:
                    label = "QUICK_BUY" if "STRONG" in high_rec else "ALIGNED_BUY"
                    direction = "CALL"
                    signal = "BUY"
                elif all_sell:
                    label = "QUICK_SELL" if "STRONG" in high_rec else "ALIGNED_SELL"
                    direction = "PUT"
                    signal = "SELL"

            df_ref = pd.DataFrame()
            ref_tf = "1h"
            if h_tf: ref_tf = h_tf
            elif m_tf: ref_tf = m_tf
            elif s_tf: ref_tf = s_tf

            df_ref = asyncio.run_coroutine_threadsafe(fetch_candles(symbol, ref_tf), manager.loop).result()
            indicators = get_ta_indicators(symbol, ref_tf)
            trend, momentum, volatility, structure = self._calculate_scores(symbol, indicators, df_ref)

            confidence = 80 if signal != "WAIT" else 0
            if signal != "WAIT" and "STRONG" in str(active_recs): confidence = 90

            # Echo Forecast Intelligence (Dynamic Source selection)
            # 1-TF: Use that TF | 2-TF: Use Smallest | 3-TF: Use Middle
            echo_tf = "5m"
            if s_tf and m_tf and h_tf: echo_tf = m_tf # 3-TF: Middle
            elif s_tf and m_tf: echo_tf = s_tf # 2-TF: Smallest
            elif s_tf: echo_tf = s_tf # 1-TF: Current
            elif m_tf: echo_tf = m_tf
            elif h_tf: echo_tf = h_tf

            df_echo = asyncio.run_coroutine_threadsafe(fetch_candles(symbol, echo_tf), manager.loop).result()
            fcast_prices, correlation = calculate_echo_forecast(df_echo, projection='Outcome')
            fcast_data = {'signals': {'small': rec_small, 'mid': rec_mid, 'high': rec_high}}
            if fcast_prices:
                fcast_final = fcast_prices[-1]
                if signal == 'BUY' and fcast_final > df_echo['close'].iloc[-1]:
                    confidence = min(100, confidence + (correlation * 10))
                elif signal == 'SELL' and fcast_final < df_echo['close'].iloc[-1]:
                    confidence = min(100, confidence + (correlation * 10))

                fcast_data.update({
                    'high': max(fcast_prices), 'low': min(fcast_prices),
                    'final': fcast_final, 'correlation': correlation,
                    'forecast_prices': fcast_prices
                })

            df_ltf = asyncio.run_coroutine_threadsafe(fetch_candles(symbol, s_tf or "1m"), manager.loop).result()
            # Strategy 7 Smart Expiry (Remove EOP)
            expiry, _ = predict_expiry(symbol, 'strategy_7', 1, 60, confidence, fcast_data, df_ltf, direction=direction)

            atr_val = 0
            tp_price, sl_price, rr = None, None, 0
            if not df_ref.empty:
                atr_val = ta.volatility.AverageTrueRange(df_ref['high'], df_ref['low'], df_ref['close']).average_true_range().iloc[-1]
                price = df_ref['close'].iloc[-1]
                # Smart Targets utilizing Echo Forecast
                tp_price, sl_price = get_smart_targets(price, 'long' if signal == 'BUY' else 'short', atr_val, confidence, fcast_data)
                if fcast_prices:
                    # Strategy 7 RR calculation (for info only, not gating)
                    rr = calculate_structural_rr(price, fcast_prices, signal if signal != 'WAIT' else direction, atr_val)

            # Relevance Logic for Strategy 7
            desc = f"RECS: {s_tf}:{rec_small} | {m_tf}:{rec_mid} | {h_tf}:{rec_high}"
            if s_tf == "OFF": desc = f"ALIGNMENT RECS: {m_tf}:{rec_mid} | {h_tf}:{rec_high}"

            data = {
                'tp': round(tp_price, 4) if tp_price else None,
                'sl': round(sl_price, 4) if sl_price else None,
                'rr': round(float(rr), 1),
                'confidence': round(float(confidence), 1),
                'label': label,
                'direction': direction,
                'signal': signal,
                'desc': desc,
                'summary_small': rec_small,
                'summary_mid': rec_mid,
                'summary_high': rec_high,
                'expiry_min': expiry,
                'expiry_countdown': expiry * 60,
                'atr': round(atr_val, 4),
                'price': round(price, 4) if 'price' in locals() else None,
                'correlation': round(correlation, 2) if 'correlation' in locals() else 0,
                'snr_count': 0, # Strat 7 uses multi-TF alignment instead of zones
                'trend': trend, 'momentum': momentum, 'volatility': volatility, 'structure': structure,
                'fcast_data': fcast_data,
                'last_update': time.time()
            }
            self.bot.screener_data[symbol] = data
            self.bot.emit('screener_update', {'symbol': symbol, 'data': data})
            return data
        except Exception as e:
            logging.error(f"Strategy 7 analysis error for {symbol}: {e}")
            return None

    def analyze_strategy_4(self, symbol):
        """Strategy 4: SNR Breakout Reversal"""
        try:
            sd = self.bot.symbol_data.get(symbol, {})
            df1m = asyncio.run_coroutine_threadsafe(fetch_candles(symbol, "1m"), manager.loop).result()
            df5m = asyncio.run_coroutine_threadsafe(fetch_candles(symbol, "5m"), manager.loop).result()

            # Calculate generic scores for display consistency
            ind5m = get_ta_indicators(symbol, "5m")
            trend, momentum, volatility, structure = self._calculate_scores(symbol, ind5m, df5m)

            atr_val = ta.volatility.AverageTrueRange(df1m['high'], df1m['low'], df1m['close']).average_true_range().iloc[-1] if not df1m.empty else 0

            # Breakout Awareness for Strategy 4
            price_4 = df1m['close'].iloc[-1]
            snr_4 = calculate_5m_snr(df5m.to_dict('records') if not df5m.empty else [])
            near_z = None
            for z in snr_4:
                if price_4 and abs(price_4 - z['price'])/price_4 < 0.01:
                    near_z = z
                    break
            
            in_zone_str = "OUTSIDE"
            if near_z:
                in_zone = price_4 >= near_z['bottom'] and price_4 <= near_z['top']
                in_zone_str = "INSIDE" if in_zone else "NEAR"
                dist_z = ((price_4 - near_z['price']) / near_z['price']) * 100 if near_z['price'] else 0
                desc = f"{in_zone_str} {near_z['type']} Zone | Dist: {dist_z:+.2f}%"
            else:
                desc = "SCANNING 5M SNR ZONES..."

            # State-based description
            s4_state = sd.get('s4_state', {})
            if s4_state.get('in_zone'):
                desc = f"WAITING FOR BREAKOUT from {s4_state['zone_type']} ZONE"

            data = {
                'signal': "WAIT",
                'direction': "NEUTRAL",
                'label': "SNR Breakout",
                'desc': desc,
                'confidence': 75 if s4_state.get('in_zone') else 50,
                'threshold': 50,
                'expiry_min': 5,
                'expiry_countdown': 300,
                'atr': round(atr_val, 4),
                'price': round(price_4, 4),
                'snr_count': len(snr_4),
                'trend': trend, 'momentum': momentum, 'volatility': volatility, 'structure': structure,
                'fcast_data': {},
                'last_update': time.time()
            }
            self.bot.screener_data[symbol] = data
            self.bot.emit('screener_update', {'symbol': symbol, 'data': data})
            return data
        except Exception as e:
            logging.error(f"Strategy 4 screener error: {e}")
            return None

    def analyze_crossover_strategy(self, symbol, strat_num, ta_interval, htf_sec):
        ta_signal = get_ta_signal(symbol, ta_interval)
        indicators = get_ta_indicators(symbol, ta_interval)

        # Consistent scores for display
        df_ta = asyncio.run_coroutine_threadsafe(fetch_candles(symbol, ta_interval), manager.loop).result()
        trend, momentum, volatility, structure = self._calculate_scores(symbol, indicators, df_ta)

        sd = self.bot.symbol_data.get(symbol, {})
        htf_open = sd.get('htf_open')
        price = sd.get('last_tick', indicators.get('close', 0))

        direction = "NEUTRAL"
        if htf_open:
            direction = "CALL" if price > htf_open else "PUT"

        # Echo Confirmation
        df_ltf = asyncio.run_coroutine_threadsafe(fetch_candles(symbol, ta_interval), manager.loop).result()
        fcast_prices, correlation = calculate_echo_forecast(df_ltf, projection='Outcome')

        echo_conf = "WAIT"
        fcast_data = {}
        if fcast_prices:
            fcast_final = fcast_prices[-1]
            if direction == "CALL" and fcast_final > price: echo_conf = "Confirmed UP"
            elif direction == "PUT" and fcast_final < price: echo_conf = "Confirmed DOWN"
            else: echo_conf = "Not Confirmed"
            fcast_data = {
                'final': fcast_final, 'correlation': correlation, 'forecast_prices': fcast_prices,
                'direction': "CALL" if fcast_final > price else "PUT"
            }

        atr_val = ta.volatility.AverageTrueRange(df_ltf['high'], df_ltf['low'], df_ltf['close']).average_true_range().iloc[-1] if not df_ltf.empty else 0
        countdown = self._get_htf_countdown(htf_sec)

        if price > 0:
            tp_price, sl_price = get_smart_targets(price, 'long' if direction == "CALL" else 'short', atr_val, correlation*100, fcast_data)
            if fcast_prices:
                rr = calculate_structural_rr(price, fcast_prices, "BUY" if direction == "CALL" else "SELL", atr_val)

        # Strategy 1 Specific Labeling
        label = f"Strategy {strat_num}"
        if strat_num == 1:
            label = "Daily Bias Crossover"
            daily_diff = ((price - htf_open) / htf_open) * 100 if htf_open else 0
            desc = f"DAILY OPEN: {htf_open:.2f} | DIFF: {daily_diff:+.2f}% | ECHO: {echo_conf}"
        else:
            desc = f"HTF Open: {htf_open} | Echo: {echo_conf}"

        data = {
            'tp': round(tp_price, 4) if tp_price else None,
            'sl': round(sl_price, 4) if sl_price else None,
            'rr': round(float(rr), 1) if 'rr' in locals() else 0,
            'signal': ta_signal,
            'direction': direction,
            'label': label,
            'desc': desc,
            'confidence': round(float(correlation * 100), 1) if 'correlation' in locals() else 0,
            'threshold': 0,
            'expiry_min': countdown // 60,
            'expiry_countdown': countdown,
            'atr': round(atr_val, 4),
            'price': round(price, 4),
            'correlation': round(correlation, 2) if 'correlation' in locals() else 0,
            'trend': trend, 'momentum': momentum, 'volatility': volatility, 'structure': structure,
            'trend_rec': ta_signal,
            'fcast_data': fcast_data,
            'last_update': time.time()
        }
        self.bot.screener_data[symbol] = data
        self.bot.emit('screener_update', {'symbol': symbol, 'data': data})
        return data

    def analyze_strategy_8(self, symbol):
        """Strategy 8: UT Bot Alerts (1m Only)"""
        try:
            df1m = asyncio.run_coroutine_threadsafe(fetch_candles(symbol, "1m"), manager.loop).result()
            if df1m.empty: return None

            indicators = get_ta_indicators(symbol, "1m")
            ut_buy = indicators.get('ut_buy', 0)
            ut_sell = indicators.get('ut_sell', 0)
            ta_rec = get_ta_signal(symbol, "1m")
            price = df1m['close'].iloc[-1]

            # Signal Logic: Explicit UT Bot Entry + TA Support
            signal = "WAIT"
            direction = "NEUTRAL"
            if ut_buy == 1:
                direction = "CALL"
                if "BUY" in ta_rec: signal = "BUY"
            elif ut_sell == 1:
                direction = "PUT"
                if "SELL" in ta_rec: signal = "SELL"

            # Echo Confirmation (3m Fixed Expiry)
            fcast_prices, correlation = calculate_echo_forecast(df1m, projection='Outcome')
            if fcast_prices:
                fcast_data = {
                    'final': fcast_prices[-1],
                    'correlation': correlation,
                    'forecast_prices': fcast_prices,
                    'high': max(fcast_prices), 'low': min(fcast_prices),
                    'direction': "CALL" if fcast_prices[-1] > price else "PUT"
                }
            else:
                fcast_data = {}

            # Confidence based on UT Bot + Correlation
            confidence = 85 if signal != "WAIT" else 50
            if correlation > 0.7: confidence = min(100, confidence + 10)

            # Alignment check with tolerance
            expiry, is_aligned = predict_expiry(symbol, 'strategy_8', 1, 15, confidence, fcast_data, df1m, direction=direction)
            if not is_aligned:
                signal = "WAIT"

            # Smart Targets for Multiplier (Tight)
            atr_series = ta.volatility.AverageTrueRange(df1m['high'], df1m['low'], df1m['close']).average_true_range()
            atr_val = atr_series.iloc[-1] if not atr_series.empty else 0
            tp_price, sl_price = get_smart_targets(price, 'long' if direction == "CALL" else 'short', atr_val, 90, fcast_data)
            rr = calculate_structural_rr(price, fcast_prices, "BUY" if direction == "CALL" else "SELL", atr_val) if (fcast_prices and signal != "WAIT") else 0

            trend, momentum, volatility, structure = self._calculate_scores(symbol, indicators, df1m)

            ut_stop = indicators.get('ut_stop', 0)
            data = {
                'tp': round(tp_price, 4) if tp_price else None,
                'sl': round(sl_price, 4) if sl_price else None,
                'rr': round(float(rr), 1),
                'signal': signal,
                'direction': direction,
                'label': "UT Bot Alerts",
                'desc': f"Stop: {ut_stop:.2f}",
                'confidence': 85 if signal != "WAIT" else 50,
                'threshold': 80,
                'expiry_min': 3,
                'expiry_countdown': 180,
                'atr': round(atr_val, 4),
                'price': round(price, 4),
                'correlation': round(correlation, 2) if 'correlation' in locals() else 0,
                'trend': trend, 'momentum': momentum, 'volatility': volatility, 'structure': structure,
                'fcast_data': fcast_data,
                'last_update': time.time()
            }
            self.bot.screener_data[symbol] = data
            self.bot.emit('screener_update', {'symbol': symbol, 'data': data})
            return data
        except Exception as e:
            logging.error(f"Strategy 8 screener error: {e}")
            return None

    def background_loop(self):
        self.bot.log("Screener background loop started")
        with ThreadPoolExecutor(max_workers=3) as executor:
            while not self.stop_event.is_set():
                config = self.bot.config
                symbols = config.get('symbols', [])
                for symbol in symbols:
                    if self.stop_event.is_set(): break
                    executor.submit(self.update_screener, symbol, config)
                    time.sleep(1.0)
                time.sleep(10)
