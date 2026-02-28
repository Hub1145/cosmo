import pandas as pd
import ta
import numpy as np

def calculate_monte_carlo(df, steps=10, simulations=100):
    if df is None or len(df) < 20: return None
    try:
        closes = df['close'].values
        returns = np.diff(np.log(closes + 1e-9))
        if len(returns) < 1: return None
        mu, sigma = np.mean(returns), np.std(returns)
        if sigma == 0: sigma = 1e-4
        drift = mu - 0.5 * sigma**2
        last_price = closes[-1]
        shocks = np.random.normal(loc=0, scale=1, size=(steps, simulations))
        price_paths = last_price * np.exp(np.cumsum(drift + sigma * shocks, axis=0))

        step_probs = []
        step_avgs = []
        for s in range(steps):
            prices = price_paths[s, :]
            bullish_prob = len(prices[prices > last_price]) / simulations * 100
            step_probs.append(bullish_prob)
            step_avgs.append({
                'up': np.mean(prices[prices > last_price]) if any(prices > last_price) else last_price,
                'down': np.mean(prices[prices < last_price]) if any(prices < last_price) else last_price
            })

        return {
            'bullish_prob': step_probs[-1],
            'bearish_prob': 100 - step_probs[-1],
            'avg_up': step_avgs[-1]['up'],
            'avg_down': step_avgs[-1]['down'],
            'upper_dev': last_price + np.std(price_paths[-1, :]),
            'lower_dev': last_price - np.std(price_paths[-1, :]),
            'final_path': np.mean(price_paths, axis=1).tolist(),
            'step_probs': step_probs,
            'step_avgs': step_avgs
        }
    except: return None

def calculate_supertrend(df, period=10, multiplier=3):
    if len(df) < period:
        return pd.Series([0.0]*len(df), index=df.index), pd.Series([0.0]*len(df), index=df.index)

    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=period).average_true_range()
    hl2 = (df['high'] + df['low']) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    final_upperband = upperband.copy()
    final_lowerband = lowerband.copy()

    for i in range(1, len(df)):
        if upperband.iloc[i] < final_upperband.iloc[i-1] or df['close'].iloc[i-1] > final_upperband.iloc[i-1]:
            final_upperband.iloc[i] = upperband.iloc[i]
        else:
            final_upperband.iloc[i] = final_upperband.iloc[i-1]

        if lowerband.iloc[i] > final_lowerband.iloc[i-1] or df['close'].iloc[i-1] < final_lowerband.iloc[i-1]:
            final_lowerband.iloc[i] = lowerband.iloc[i]
        else:
            final_lowerband.iloc[i] = final_lowerband.iloc[i-1]

    supertrend = [0.0] * len(df)
    direction = [1] * len(df) # 1 for up, -1 for down

    for i in range(1, len(df)):
        if i == 1:
            supertrend[i] = final_upperband.iloc[i]
            direction[i] = -1
            continue
        if supertrend[i-1] == final_upperband.iloc[i-1]:
            if df['close'].iloc[i] > final_upperband.iloc[i]:
                supertrend[i] = final_lowerband.iloc[i]
                direction[i] = 1
            else:
                supertrend[i] = final_upperband.iloc[i]
                direction[i] = -1
        else:
            if df['close'].iloc[i] < final_lowerband.iloc[i]:
                supertrend[i] = final_upperband.iloc[i]
                direction[i] = -1
            else:
                supertrend[i] = final_lowerband.iloc[i]
                direction[i] = 1
    return pd.Series(supertrend, index=df.index), pd.Series(direction, index=df.index)

def calculate_fractals(df, window=2):
    if len(df) < 2 * window + 1:
        return pd.Series([False]*len(df), index=df.index), pd.Series([False]*len(df), index=df.index)
    highs = df['high']
    lows = df['low']
    is_high = [False] * len(df)
    is_low = [False] * len(df)
    for i in range(window, len(df) - window):
        try:
            if all(highs.iloc[i] > highs.iloc[i-window:i]) and all(highs.iloc[i] > highs.iloc[i+1:i+window+1]):
                is_high[i] = True
            if all(lows.iloc[i] < lows.iloc[i-window:i]) and all(lows.iloc[i] < lows.iloc[i+1:i+window+1]):
                is_low[i] = True
        except: continue
    return pd.Series(is_high, index=df.index), pd.Series(is_low, index=df.index)

def calculate_pivot_points(df, left=15, right=15):
    if len(df) < (left + right + 1):
        return pd.Series([False]*len(df), index=df.index), pd.Series([False]*len(df), index=df.index)
    highs = df['high'].values
    lows = df['low'].values
    pivot_highs = [False] * len(df)
    pivot_lows = [False] * len(df)
    for i in range(left, len(df) - right):
        val_h = highs[i]
        if np.isnan(val_h): continue
        is_h = True
        for j in range(i - left, i):
            if highs[j] > val_h:
                is_h = False; break
        if is_h:
            for j in range(i + 1, i + right + 1):
                if highs[j] >= val_h:
                    is_h = False; break
        if is_h: pivot_highs[i] = True
        val_l = lows[i]
        if np.isnan(val_l): continue
        is_l = True
        for j in range(i - left, i):
            if lows[j] < val_l:
                is_l = False; break
        if is_l:
            for j in range(i + 1, i + right + 1):
                if lows[j] <= val_l:
                    is_l = False; break
        if is_l: pivot_lows[i] = True
    return pd.Series(pivot_highs, index=df.index), pd.Series(pivot_lows, index=df.index)

def calculate_order_blocks(df, lookback=100):
    if len(df) < lookback: return []
    obs = []
    for i in range(len(df) - 5, 5, -1):
        if i < 10: break
        avg_body = abs(df['close'].iloc[i-10:i] - df['open'].iloc[i-10:i]).mean()
        body = abs(df['close'].iloc[i] - df['open'].iloc[i])
        if body > 2 * avg_body:
            is_bullish_impulse = df['close'].iloc[i] > df['open'].iloc[i]
            for j in range(i-1, i-6, -1):
                if is_bullish_impulse and df['close'].iloc[j] < df['open'].iloc[j]:
                    obs.append({'price': df['low'].iloc[j], 'high': df['high'].iloc[j], 'type': 'Bullish OB', 'epoch': df['epoch'].iloc[j]})
                    break
                elif not is_bullish_impulse and df['close'].iloc[j] > df['open'].iloc[j]:
                    obs.append({'price': df['high'].iloc[j], 'low': df['low'].iloc[j], 'type': 'Bearish OB', 'epoch': df['epoch'].iloc[j]})
                    break
        if len(obs) >= 5: break
    return obs

def calculate_fvg(df, lookback=50):
    if len(df) < 3: return []
    fvgs = []
    for i in range(len(df) - 1, len(df) - lookback, -1):
        if i < 2: break
        if df['high'].iloc[i-2] < df['low'].iloc[i]:
            fvgs.append({'top': df['low'].iloc[i], 'bottom': df['high'].iloc[i-2], 'type': 'Bullish FVG', 'epoch': df['epoch'].iloc[i-1]})
        elif df['low'].iloc[i-2] > df['high'].iloc[i]:
            fvgs.append({'top': df['low'].iloc[i-2], 'bottom': df['high'].iloc[i], 'type': 'Bearish FVG', 'epoch': df['epoch'].iloc[i-1]})
        if len(fvgs) >= 10: break
    return fvgs

def detect_macd_divergence(df, window=20):
    if len(df) < window + 10: return 0
    macd_ind = ta.trend.MACD(df['close'])
    macd = macd_ind.macd()
    p_prev_low = df['close'].iloc[-2*window:-window].min()
    m_prev_low = macd.iloc[-2*window:-window].min()
    if df['close'].iloc[-1] < p_prev_low and macd.iloc[-1] > m_prev_low:
        return 1
    p_prev_high = df['close'].iloc[-2*window:-window].max()
    m_prev_high = macd.iloc[-2*window:-window].max()
    if df['close'].iloc[-1] > p_prev_high and macd.iloc[-1] < m_prev_high:
        return -1
    return 0

def check_price_action_patterns(candles):
    if len(candles) < 2: return None
    curr, prev = candles[-1], candles[-2]
    body = abs(curr['close'] - curr['open'])
    upper_wick = curr['high'] - max(curr['open'], curr['close'])
    lower_wick = min(curr['open'], curr['close']) - curr['low']
    total_range = curr['high'] - curr['low']
    if total_range == 0: return None
    if body > (total_range * 0.9): return "marubozu"
    if body < (total_range * 0.35):
        if lower_wick > (total_range * 0.6): return "bullish_pin"
        if upper_wick > (total_range * 0.6): return "bearish_pin"
    prev_body = abs(prev['close'] - prev['open'])
    if body > prev_body:
        if curr['close'] > curr['open'] and prev['close'] < prev['open']:
            if curr['close'] >= prev['open'] and curr['open'] <= prev['close']: return "bullish_engulfing"
        if curr['close'] < curr['open'] and prev['close'] > prev['open']:
            if curr['close'] <= prev['open'] and curr['open'] >= prev['close']: return "bearish_engulfing"
    if body < prev_body * 0.5:
        if max(curr['open'], curr['close']) <= max(prev['open'], prev['close']) and \
           min(curr['open'], curr['close']) >= min(prev['open'], prev['close']):
            return "bullish_harami" if curr['close'] > curr['open'] else "bearish_harami"
    if abs(curr['high'] - prev['high']) < (total_range * 0.05) and curr['high'] > max(curr['open'], curr['close']): return "tweezer_top"
    if abs(curr['low'] - prev['low']) < (total_range * 0.05) and curr['low'] < min(curr['open'], curr['close']): return "tweezer_bottom"
    if body < (total_range * 0.1): return "doji"
    return None

def calculate_adr(daily_candles, window=14):
    if len(daily_candles) < window: return 0
    ranges = [c['high'] - c['low'] for c in daily_candles[-window:]]
    return sum(ranges) / len(ranges)

def calculate_snr_zones(symbol, sd, granularity=None, active_strategy=None):
    if not sd: return []
    if granularity is None:
        if active_strategy == 'strategy_1': granularity = 86400
        elif active_strategy == 'strategy_2': granularity = 3600
        elif active_strategy == 'strategy_3': granularity = 900
        else: granularity = 3600
    candles = []
    if granularity == 3600:  candles = sd.get('htf_candles', [])
    elif granularity == 900: candles = sd.get('m15_candles', [])
    elif granularity == 300: candles = sd.get('m5_candles', [])
    elif granularity == 86400: candles = sd.get('daily_candles', [])
    LEFT, RIGHT = 15, 15
    if len(candles) < (LEFT + RIGHT + 1): return sd.get('snr_zones', [])
    candles = candles[-200:]
    n = len(candles)
    highs, lows, closes = [c['high'] for c in candles], [c['low'] for c in candles], [c['close'] for c in candles]
    pivot_highs, pivot_lows = [None] * n, [None] * n
    for i in range(LEFT, n - RIGHT):
        window_h = highs[i - LEFT : i + RIGHT + 1]
        if highs[i] == max(window_h): pivot_highs[i] = highs[i]
        window_l = lows[i - LEFT : i + RIGHT + 1]
        if lows[i] == min(window_l): pivot_lows[i] = lows[i]
    res_levels, sup_levels = [], []
    for i in range(n):
        if pivot_highs[i] is not None: res_levels.append({'price': pivot_highs[i], 'idx': i})
        if pivot_lows[i] is not None: sup_levels.append({'price': pivot_lows[i], 'idx': i})
    avg_price = sum(closes) / len(closes) if closes else 1
    threshold = avg_price * 0.0005
    def cluster_levels(levels, zone_type):
        clusters = []
        for lv in levels:
            found = False
            for c in clusters:
                if abs(lv['price'] - c['price']) < threshold:
                    c['prices'].append(lv['price']); c['price'] = sum(c['prices']) / len(c['prices']); c['touches'] += 1; found = True; break
            if not found: clusters.append({'price': lv['price'], 'prices': [lv['price']], 'touches': 1, 'type': zone_type, 'is_flip': False})
        return clusters
    r_clusters, s_clusters = cluster_levels(res_levels, 'R'), cluster_levels(sup_levels, 'S')
    current_price = closes[-1]
    all_zones = []
    for c in r_clusters:
        is_flip = c['price'] < current_price
        all_zones.append({'price': c['price'], 'touches': c['touches'], 'is_flip': is_flip, 'type': 'Flip' if is_flip else 'R', 'total_lifetime_touches': c['touches']})
    for c in s_clusters:
        is_flip = c['price'] > current_price
        all_zones.append({'price': c['price'], 'touches': c['touches'], 'is_flip': is_flip, 'type': 'Flip' if is_flip else 'S', 'total_lifetime_touches': c['touches']})
    all_zones.sort(key=lambda x: x['touches'], reverse=True)
    all_zones = sorted(all_zones[:10], key=lambda x: abs(x['price'] - current_price))
    return all_zones[:5]

def calculate_5m_snr(m5_candles):
    LEFT, RIGHT = 15, 15
    if len(m5_candles) < (LEFT + RIGHT + 1): return []
    n = len(m5_candles)
    highs, lows = [c['high'] for c in m5_candles], [c['low'] for c in m5_candles]
    zones = []
    for i in range(LEFT, n - RIGHT):
        c = m5_candles[i]
        window_h = highs[i - LEFT : i + RIGHT + 1]
        if highs[i] == max(window_h):
            body_top = max(c['open'], c['close'])
            zones.append({'price': highs[i], 'top': highs[i], 'bottom': (highs[i] + body_top) / 2, 'type': 'R', 'epoch': c.get('epoch', 0)})
        window_l = lows[i - LEFT : i + RIGHT + 1]
        if lows[i] == min(window_l):
            body_bottom = min(c['open'], c['close'])
            zones.append({'price': lows[i], 'bottom': lows[i], 'top': (lows[i] + body_bottom) / 2, 'type': 'S', 'epoch': c.get('epoch', 0)})
    return zones[-10:]

def calculate_ut_bot(df, key_value=1, atr_period=10):
    highs, lows, closes = df['high'].values.astype(float), df['low'].values.astype(float), df['close'].values.astype(float)
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=atr_period).average_true_range().values
    n_loss = key_value * atr
    src = closes
    trailing_stop = np.zeros(len(df)); trailing_stop[0] = src[0]
    for i in range(1, len(df)):
        if np.isnan(n_loss[i]): trailing_stop[i] = src[i]; continue
        if src[i] > trailing_stop[i-1] and src[i-1] > trailing_stop[i-1]: trailing_stop[i] = max(trailing_stop[i-1], src[i] - n_loss[i])
        elif src[i] < trailing_stop[i-1] and src[i-1] < trailing_stop[i-1]: trailing_stop[i] = min(trailing_stop[i-1], src[i] + n_loss[i])
        elif src[i] > trailing_stop[i-1]: trailing_stop[i] = src[i] - n_loss[i]
        else: trailing_stop[i] = src[i] + n_loss[i]
    pos = np.zeros(len(df))
    for i in range(1, len(df)):
        if src[i-1] < trailing_stop[i-1] and src[i] > trailing_stop[i-1]: pos[i] = 1
        elif src[i-1] > trailing_stop[i-1] and src[i] < trailing_stop[i-1]: pos[i] = -1
        else: pos[i] = pos[i-1]
    buy_signals, sell_signals = np.zeros(len(df), dtype=int), np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        above = (src[i-1] <= trailing_stop[i-1]) and (src[i] > trailing_stop[i])
        below = (src[i-1] >= trailing_stop[i-1]) and (src[i] < trailing_stop[i])
        if (src[i] > trailing_stop[i]) and above: buy_signals[i] = 1
        if (src[i] < trailing_stop[i]) and below: sell_signals[i] = 1
    return trailing_stop, pos, buy_signals, sell_signals

def calculate_stoch_rsi(close, window=14, smooth_k=3, smooth_d=3):
    rsi = ta.momentum.RSIIndicator(close, window=window).rsi()
    rsi_low, rsi_high = rsi.rolling(window=window).min(), rsi.rolling(window=window).max()
    stoch_rsi = (rsi - rsi_low) / (rsi_high - rsi_low)
    k = stoch_rsi.rolling(window=smooth_k).mean() * 100
    d = k.rolling(window=smooth_d).mean()
    return k, d

def score_reversal_pattern(symbol, pattern, candles):
    if not candles: return 0
    c = candles[-1]
    prev = candles[-2] if len(candles) > 1 else None
    score = 0
    body, total_range = abs(c['close'] - c['open']), c['high'] - c['low']
    if total_range == 0: return 0
    upper_wick, lower_wick = c['high'] - max(c['open'], c['close']), min(c['open'], c['close']) - c['low']
    max_wick = max(upper_wick, lower_wick)
    if body > 0 and (max_wick / body) >= 2: score += 1
    elif body == 0: score += 1
    if pattern.startswith('bullish'):
        if c['close'] >= (c['low'] + total_range * 0.75): score += 1
    elif pattern.startswith('bearish'):
        if c['close'] <= (c['low'] + total_range * 0.25): score += 1
    elif pattern == 'doji': score += 1
    if prev:
        prev_body, prev_range = abs(prev['close'] - prev['open']), prev['high'] - prev['low']
        if prev_range > 0 and (prev_body / prev_range) > 0.6: score += 1
    return score

def get_smart_multiplier(atr_pct, base_multiplier=100):
    if atr_pct == 0: return base_multiplier
    scale = 0.001 / atr_pct
    return int(max(10, min(500, base_multiplier * scale)))

def predict_expiry(symbol, strategy_key, ltf_min, htf_min, confidence, fcast_data, df_ltf, direction='NEUTRAL'):
    is_aligned = True
    if strategy_key == 'strategy_7':
        signals = fcast_data.get('signals', {})
        rec_small, rec_mid, rec_high = str(signals.get('small', "OFF")), str(signals.get('mid', "OFF")), str(signals.get('high', "OFF"))
        if rec_high != "OFF": return (20 if "STRONG" in rec_high else 60), True
        if "STRONG" in rec_mid: return max(1, min(4, int(1 + (confidence / 33)))), True
        if ("BUY" in rec_mid or "SELL" in rec_mid) and ("BUY" in rec_small or "SELL" in rec_small): return 20, True
        return 5, True

    if strategy_key in ['strategy_5', 'strategy_6', 'strategy_9']:
        mc = fcast_data.get('mc_data', {})
        echo_prices = fcast_data.get('forecast_prices', [])
        if mc and echo_prices:
            last_price = df_ltf['close'].iloc[-1] if df_ltf is not None and not df_ltf.empty else 0
            probs = mc.get('step_probs', [])
            scores = []
            for n in range(min(len(probs), len(echo_prices))):
                echo_val = echo_prices[n]
                echo_score = 1.0
                if direction in ['CALL', 'BUY']:
                    if echo_val > last_price: echo_score = 1.0 + (echo_val - last_price)/last_price if last_price else 1.0
                    else: echo_score = 0.5
                else:
                    if echo_val < last_price: echo_score = 1.0 + (last_price - echo_val)/last_price if last_price else 1.0
                    else: echo_score = 0.5
                inflection = False
                if 0 < n < len(echo_prices)-1:
                    if direction in ['CALL', 'BUY'] and echo_prices[n] > echo_prices[n-1] and echo_prices[n] > echo_prices[n+1]: inflection = True
                    if direction in ['PUT', 'SELL'] and echo_prices[n] < echo_prices[n-1] and echo_prices[n] < echo_prices[n+1]: inflection = True
                if inflection: echo_score *= 0.5
                mc_prob = probs[n] if direction in ['CALL', 'BUY'] else (100 - probs[n])
                scores.append(echo_score * (mc_prob / 100.0))
            if scores:
                best_n = scores.index(max(scores)) + 1
                return best_n, True
    base_expiry = 5
    if ltf_min: base_expiry = ltf_min * 3
    return max(1, base_expiry), is_aligned

def calculate_structural_rr(current_price: float, forecast_prices: list, direction: str, atr: float = 0):
    if not forecast_prices: return 1.0
    forecast_max, forecast_min = max(forecast_prices), min(forecast_prices)
    if direction.upper() in ["BUY", "CALL", "LONG"]: reward, risk = forecast_max - current_price, current_price - forecast_min
    else: reward, risk = current_price - forecast_min, forecast_max - current_price
    final_risk = max(risk, atr)
    return 10.0 if final_risk <= 0 else reward / final_risk

def get_smart_targets(entry_price, side, atr, confidence, fcast_data=None):
    if atr == 0: return None, None
    is_long, sl_dist = side == 'long', 1.5 * atr
    mc = fcast_data.get('mc_data', {}) if fcast_data else {}
    if mc:
        sl_dist = abs(entry_price - (mc['lower_dev'] if is_long else mc['upper_dev']))
        sl_dist = max(sl_dist, 1.2 * atr)
    tp_dist = sl_dist * (2 + (3 * (confidence / 100)))
    if fcast_data and fcast_data.get('correlation', 0) > 0.6:
        tp_dist = max(tp_dist, abs(entry_price - (fcast_data.get('high', entry_price) if is_long else fcast_data.get('low', entry_price))))
    if mc:
        tp_dist = max(tp_dist, abs(entry_price - (mc.get('avg_up', entry_price) if is_long else mc.get('avg_down', entry_price))))
    return (entry_price + tp_dist) if is_long else (entry_price - tp_dist), (entry_price - sl_dist) if is_long else (entry_price + sl_dist)

def calculate_echo_forecast(df, eval_window=50, forecast_window=50, fmode='Similarity', projection='Pattern'):
    if df is None or len(df) < (eval_window + forecast_window * 2 + 1): return None, 0
    src, deltas = df['close'].values, df['close'].diff().values
    ref, best_val, best_k = src[-forecast_window:], (-1.0 if fmode == 'Similarity' else 1.0), 0
    for i in range(eval_window):
        match_end_idx = len(src) - forecast_window - i
        match_start_idx = match_end_idx - forecast_window
        if match_start_idx < 0: break
        b = src[match_start_idx:match_end_idx]
        std_ref, std_b = np.std(ref), np.std(b)
        r = np.corrcoef(ref, b)[0, 1] if std_ref != 0 and std_b != 0 else 0
        if np.isnan(r): r = 0
        if (fmode == 'Similarity' and r > best_val) or (fmode != 'Similarity' and r < best_val): best_val, best_k = r, i
    match_start = len(src) - forecast_window*2 - best_k if projection == 'Pattern' else len(src) - forecast_window - best_k
    forecast_deltas = deltas[match_start : match_start + forecast_window]
    current_price, forecast_prices, temp_price = src[-1], [], src[-1]
    for d in forecast_deltas:
        if np.isnan(d): d = 0
        temp_price += d; forecast_prices.append(temp_price)
    return forecast_prices, best_val
