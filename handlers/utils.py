import pandas as pd
import ta
import numpy as np

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
    """
    Identifies LuxAlgo-style pivot highs and lows.
    """
    if len(df) < (left + right + 1):
        return pd.Series([False]*len(df), index=df.index), pd.Series([False]*len(df), index=df.index)

    highs = df['high'].values
    lows = df['low'].values

    pivot_highs = [False] * len(df)
    pivot_lows = [False] * len(df)

    for i in range(left, len(df) - right):
        # Pivot High
        val_h = highs[i]
        if np.isnan(val_h): continue
        is_h = True
        for j in range(i - left, i):
            if highs[j] > val_h:
                is_h = False
                break
        if is_h:
            for j in range(i + 1, i + right + 1):
                if highs[j] >= val_h:
                    is_h = False
                    break
        if is_h: pivot_highs[i] = True

        # Pivot Low
        val_l = lows[i]
        if np.isnan(val_l): continue
        is_l = True
        for j in range(i - left, i):
            if lows[j] < val_l:
                is_l = False
                break
        if is_l:
            for j in range(i + 1, i + right + 1):
                if lows[j] <= val_l:
                    is_l = False
                    break
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
    """
    LuxAlgo-faithful SNR Zone Detection.
    Port of: highUsePivot = fixnan(pivothigh(15, 15))
              lowUsePivot  = fixnan(pivotlow (15, 15))
    
    Zones are persistent: a level is held until a new pivot replaces it.
    Multi-touch clustering is applied to filter noisy single-touch pivots.
    Returns a list of zone dicts compatible with all callers.
    """
    if not sd: return []

    # Select the correct candle pool based on granularity / strategy
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

    # Need at least LEFT+RIGHT+1 candles for the pivot window
    LEFT  = 15
    RIGHT = 15
    MIN_REQUIRED = LEFT + RIGHT + 1

    if len(candles) < MIN_REQUIRED:
        return sd.get('snr_zones', [])

    candles = candles[-200:]   # Use up to 200 most recent candles
    n = len(candles)

    highs  = [c['high']  for c in candles]
    lows   = [c['low']   for c in candles]
    closes = [c['close'] for c in candles]

    # ── Step 1: LuxAlgo pivot detection (left=15, right=15) ───────────
    # pivothigh(leftBars, rightBars): bar i is a pivot high if
    #   highs[i] is strictly the max over [i-LEFT .. i+RIGHT]
    pivot_highs = [None] * n
    pivot_lows  = [None] * n

    for i in range(LEFT, n - RIGHT):
        window_h = highs[i - LEFT : i + RIGHT + 1]
        if highs[i] == max(window_h):
            pivot_highs[i] = highs[i]

        window_l = lows[i - LEFT : i + RIGHT + 1]
        if lows[i] == min(window_l):
            pivot_lows[i] = lows[i]

    # ── Step 2: fixnan = forward-fill (last seen pivot holds until replaced)
    res_ff = None   # current forward-filled resistance level
    sup_ff = None   # current forward-filled support level
    res_levels = [] # list of all resistance pivots (price, candle index)
    sup_levels = [] # list of all support pivots

    for i in range(n):
        if pivot_highs[i] is not None:
            res_ff = pivot_highs[i]
            res_levels.append({'price': res_ff, 'idx': i})
        if pivot_lows[i] is not None:
            sup_ff = pivot_lows[i]
            sup_levels.append({'price': sup_ff, 'idx': i})

    # ── Step 3: Cluster nearby pivots (within 0.05% of each other) ────
    avg_price = sum(closes) / len(closes) if closes else 1
    threshold = avg_price * 0.0005

    def cluster_levels(levels, zone_type):
        clusters = []
        for lv in levels:
            found = False
            for c in clusters:
                if abs(lv['price'] - c['price']) < threshold:
                    # Update cluster average price
                    c['prices'].append(lv['price'])
                    c['price'] = sum(c['prices']) / len(c['prices'])
                    c['touches'] += 1
                    found = True
                    break
            if not found:
                clusters.append({
                    'price': lv['price'],
                    'prices': [lv['price']],
                    'touches': 1,
                    'type': zone_type,
                    'is_flip': False
                })
        return clusters

    r_clusters = cluster_levels(res_levels, 'R')
    s_clusters = cluster_levels(sup_levels, 'S')

    # ── Step 4: Mark flip zones (price crossed from S to R or vice versa)
    current_price = closes[-1]
    all_zones = []

    for c in r_clusters:
        is_flip = c['price'] < current_price   # resistance below price = flipped to support
        all_zones.append({
            'price': c['price'],
            'touches': c['touches'],
            'is_flip': is_flip,
            'type': 'Flip' if is_flip else 'R',
            'total_lifetime_touches': c['touches']
        })

    for c in s_clusters:
        is_flip = c['price'] > current_price   # support above price = flipped to resistance
        all_zones.append({
            'price': c['price'],
            'touches': c['touches'],
            'is_flip': is_flip,
            'type': 'Flip' if is_flip else 'S',
            'total_lifetime_touches': c['touches']
        })

    # ── Step 5: Merge with historical lifetime touch counts ───────────
    old_zones = sd.get('snr_zones', [])
    for z in all_zones:
        for oz in old_zones:
            if oz.get('price') and abs(z['price'] - oz['price']) / oz['price'] < 0.001:
                z['total_lifetime_touches'] = max(
                    z['total_lifetime_touches'],
                    oz.get('total_lifetime_touches', 0)
                )
                break

    # ── Step 6: Sort by strength (touches), return closest 5 to price ─
    all_zones.sort(key=lambda x: x['touches'], reverse=True)
    # Keep no more than 10 and prioritise proximity to current price
    all_zones = sorted(all_zones[:10], key=lambda x: abs(x['price'] - current_price))
    return all_zones[:5]


def calculate_5m_snr(m5_candles):
    """
    Strategy 4: 5m SNR Zones using LuxAlgo Pivot Logic (left=10, right=10).
    Zones are defined from the wick extreme to the midpoint of the wick vs body.
    Matches the original LuxAlgo indicator's visual zone rendering.
    """
    LEFT  = 10
    RIGHT = 10
    MIN_REQUIRED = LEFT + RIGHT + 1

    if len(m5_candles) < MIN_REQUIRED:
        return []

    n = len(m5_candles)
    highs = [c['high'] for c in m5_candles]
    lows  = [c['low']  for c in m5_candles]

    zones = []
    for i in range(LEFT, n - RIGHT):
        c = m5_candles[i]

        # Pivot High: Resistance zone
        window_h = highs[i - LEFT : i + RIGHT + 1]
        if highs[i] == max(window_h):
            body_top = max(c['open'], c['close'])
            # Resistance zone: wick tip down to midpoint of upper wick
            mid_upper_wick = (highs[i] + body_top) / 2
            zones.append({
                'price':  highs[i],
                'top':    highs[i],
                'bottom': mid_upper_wick,
                'type':   'R',
                'epoch':  c.get('epoch', 0)
            })

        # Pivot Low: Support zone
        window_l = lows[i - LEFT : i + RIGHT + 1]
        if lows[i] == min(window_l):
            body_bottom = min(c['open'], c['close'])
            # Support zone: wick tip up to midpoint of lower wick
            mid_lower_wick = (lows[i] + body_bottom) / 2
            zones.append({
                'price':  lows[i],
                'bottom': lows[i],
                'top':    mid_lower_wick,
                'type':   'S',
                'epoch':  c.get('epoch', 0)
            })

    # Keep only the most recent 10 zones (closest to current bar)
    return zones[-10:]

def calculate_ut_bot(df, key_value=1, atr_period=10):
    """
    UT Bot Alerts Logic (ATR Trailing Stop).
    Ported from PineScript v4.
    """
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)
    closes = df['close'].values.astype(float)
    
    # ATR calculation (Wilder Smoothing) using 'ta' library
    atr_series = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=atr_period).average_true_range()
    atr = atr_series.values
    n_loss = key_value * atr
    
    src = closes
    trailing_stop = np.zeros(len(df))
    trailing_stop[0] = src[0]
    
    for i in range(1, len(df)):
        if np.isnan(n_loss[i]):
            trailing_stop[i] = src[i]
            continue
            
        curr_src = src[i]
        prev_src = src[i-1]
        prev_stop = trailing_stop[i-1]
        
        if curr_src > prev_stop and prev_src > prev_stop:
            trailing_stop[i] = max(prev_stop, curr_src - n_loss[i])
        elif curr_src < prev_stop and prev_src < prev_stop:
            trailing_stop[i] = min(prev_stop, curr_src + n_loss[i])
        elif curr_src > prev_stop:
            trailing_stop[i] = curr_src - n_loss[i]
        else:
            trailing_stop[i] = curr_src + n_loss[i]
            
    # Trend Position
    pos = np.zeros(len(df))
    for i in range(1, len(df)):
        if src[i-1] < trailing_stop[i-1] and src[i] > trailing_stop[i-1]:
            pos[i] = 1
        elif src[i-1] > trailing_stop[i-1] and src[i] < trailing_stop[i-1]:
            pos[i] = -1
        else:
            pos[i] = pos[i-1]
            
    # Buy/Sell Signals (PineScript crossover logic)
    # ema(src,1) is src
    # above = crossover(src, xATRTrailingStop)
    # below = crossover(xATRTrailingStop, src)
    # buy = src > xATRTrailingStop and above
    # sell = src < xATRTrailingStop and below
    buy_signals = np.zeros(len(df), dtype=int)
    sell_signals = np.zeros(len(df), dtype=int)
    
    for i in range(1, len(df)):
        above = (src[i-1] <= trailing_stop[i-1]) and (src[i] > trailing_stop[i])
        below = (src[i-1] >= trailing_stop[i-1]) and (src[i] < trailing_stop[i])
        
        if (src[i] > trailing_stop[i]) and above:
            buy_signals[i] = 1
        if (src[i] < trailing_stop[i]) and below:
            sell_signals[i] = 1
            
    return trailing_stop, pos, buy_signals, sell_signals

def calculate_5m_snr(m5_candles):
    """
    Strategy 4: 5m SNR Zones using LuxAlgo Pivot Logic.
    Zones are defined from the High/Low to the midpoint of the wick (High/Low to Body).
    """
    if len(m5_candles) < 40:
        return []

    # Get recent pivot highs/lows on 5m (LuxAlgo style 15-15)
    df = pd.DataFrame(m5_candles)
    highs, lows = calculate_pivot_points(df, left=15, right=15)

    zones = []
    for i in range(len(df)):
        c = m5_candles[i]
        if highs.iloc[i]:
            body_top = max(c['open'], c['close'])
            # Resistance zone: High to midpoint of upper wick
            zones.append({
                'price': c['high'],
                'top': c['high'],
                'bottom': (c['high'] + body_top) / 2,
                'type': 'R',
                'epoch': c['epoch']
            })
        if lows.iloc[i]:
            body_bottom = min(c['open'], c['close'])
            # Support zone: Low to midpoint of lower wick
            zones.append({
                'price': c['low'],
                'bottom': c['low'],
                'top': (c['low'] + body_bottom) / 2,
                'type': 'S',
                'epoch': c['epoch']
            })

    # Keep only the last 10 unique zones
    return zones[-10:]

def calculate_stoch_rsi(close, window=14, smooth_k=3, smooth_d=3):
    rsi = ta.momentum.RSIIndicator(close, window=window).rsi()
    rsi_low = rsi.rolling(window=window).min()
    rsi_high = rsi.rolling(window=window).max()
    stoch_rsi = (rsi - rsi_low) / (rsi_high - rsi_low)
    k = stoch_rsi.rolling(window=smooth_k).mean() * 100
    d = k.rolling(window=smooth_d).mean()
    return k, d

def score_reversal_pattern(symbol, pattern, candles):
    if not candles: return 0
    c = candles[-1]
    prev = candles[-2] if len(candles) > 1 else None

    score = 0
    body = abs(c['close'] - c['open'])
    total_range = c['high'] - c['low']
    if total_range == 0: return 0

    # 1. Wick-to-body ratio (>2:1)
    upper_wick = c['high'] - max(c['open'], c['close'])
    lower_wick = min(c['open'], c['close']) - c['low']

    max_wick = max(upper_wick, lower_wick)
    if body > 0 and (max_wick / body) >= 2: score += 1
    elif body == 0: score += 1

    # 2. Close position within candle (top/bottom 25%)
    if pattern.startswith('bullish'):
        if c['close'] >= (c['low'] + total_range * 0.75): score += 1
    elif pattern.startswith('bearish'):
        if c['close'] <= (c['low'] + total_range * 0.25): score += 1
    elif pattern == 'doji': score += 1

    # 3. Prior candle strongly directional
    if prev:
        prev_body = abs(prev['close'] - prev['open'])
        prev_range = prev['high'] - prev['low']
        if prev_range > 0 and (prev_body / prev_range) > 0.6: score += 1

    return score

def get_smart_multiplier(atr_pct, base_multiplier=100):
    """
    Scale multiplier based on relative volatility (ATR as % of price).
    Low Volatility -> Higher Multiplier.
    High Volatility -> Lower Multiplier.
    """
    # Typical ATR% for indices might be 0.05% to 0.5%
    # If ATR% is 0.1%, use base.
    # If ATR% is 0.5%, use base/2.
    # If ATR% is 0.02%, use base*2.

    if atr_pct == 0: return base_multiplier

    # Target volatility index: 0.1% (0.001)
    scale = 0.001 / atr_pct
    multiplier = base_multiplier * scale

    # Constrain to sensible limits (e.g. 10x to 500x)
    return int(max(10, min(500, multiplier)))

def predict_expiry(symbol, strategy_key, ltf_min, htf_min, confidence, fcast_data, df_ltf, direction='NEUTRAL'):
    """
    Expert Intelligence Expiry Engine (Enhanced with Profitable Arrival & Alignment Logic).
    Returns (expiry_candles, is_aligned)
    """
    # 1. Base Intelligence from ATR Speed
    atr = 0
    curr_price = 0
    if df_ltf is not None and len(df_ltf) >= 14:
        atr_series = ta.volatility.AverageTrueRange(df_ltf['high'], df_ltf['low'], df_ltf['close'], window=14).average_true_range()
        atr = atr_series.iloc[-1]
        curr_price = df_ltf['close'].iloc[-1]

    base_expiry = 5
    if ltf_min: base_expiry = ltf_min * 3

    is_aligned = True # Default to true for fallback

    # 2. Echo Forecast Arrival & Alignment Logic
    if fcast_data and 'forecast_prices' in fcast_data and fcast_data.get('correlation', 0) > 0.4:
        prices = fcast_data['forecast_prices']
        
        # A. Profitable Arrival Detection
        # Identify the first candle in the forecast that is profitable
        profitable_index = -1
        for idx, p in enumerate(prices):
            if direction in ['CALL', 'BUY'] and p > curr_price:
                profitable_index = idx + 1
                break
            elif direction in ['PUT', 'SELL'] and p < curr_price:
                profitable_index = idx + 1
                break
        
        # B. Alignment check at specific horizons (1m and 5m if available)
        # Only set is_aligned to false if the forecast is STRONGLY against us.
        # Minor pullbacks are acceptable as long as the profitable arrival exists.
        if len(prices) >= 5:
            at_5 = prices[4]
            # If price is > 1 ATR against us at 5th candle, then it's unaligned.
            tolerance = atr if atr > 0 else (curr_price * 0.0001)
            if direction in ['CALL', 'BUY'] and at_5 < (curr_price - tolerance): is_aligned = False
            elif direction in ['PUT', 'SELL'] and at_5 > (curr_price + tolerance): is_aligned = False

        if profitable_index != -1:
            base_expiry = profitable_index
            # Smart Dynamic Cap: Limit to htf_min or 60 minutes
            max_cap = htf_min if htf_min else 60
            base_expiry = min(base_expiry, max_cap)
            
            return max(1, base_expiry), is_aligned
        else:
            # Fallback to extreme point but mark as unaligned if never profitable
            is_aligned = False
            try:
                if direction in ['CALL', 'BUY']:
                    base_expiry = prices.index(max(prices)) + 1
                elif direction in ['PUT', 'SELL']:
                    base_expiry = prices.index(min(prices)) + 1
                
                max_cap = htf_min if htf_min else 60
                base_expiry = min(base_expiry, max_cap)
            except:
                pass
            return max(1, base_expiry), is_aligned

    # 3. Fallback Strategy-Specific Logic
    if strategy_key in ['strategy_5', 'strategy_6']:
        # If no forecast, use ATR-based target distance or simple 5-min default
        target_candles = 5 - int(4 * (confidence / 100))
        base_expiry = max(1, min(15, target_candles))

    elif strategy_key == 'strategy_7':
        # Logic for Strategy 7 mapping
        base_expiry = 5 
        # (Simplified for now, Strat 7 usually uses its own logic)

    return max(1, base_expiry), is_aligned

def calculate_structural_rr(current_price: float, forecast_prices: list, direction: str, atr: float = 0):
    """
    Calculates the Reward/Risk ratio based on the projected structural path.
    Reward = Distance to the projected extreme in signal direction.
    Risk = Distance to the projected opposite extreme (potential pullback/stop).
    Uses ATR as a risk floor to ensure robust calculation.
    """
    if not forecast_prices:
        return 1.0

    forecast_max = max(forecast_prices)
    forecast_min = min(forecast_prices)

    if direction.upper() in ["BUY", "CALL", "LONG"]:
        reward = forecast_max - current_price
        risk = current_price - forecast_min
    else:
        reward = current_price - forecast_min
        risk = forecast_max - current_price

    # Use ATR as risk floor (1.0x ATR minimum risk)
    final_risk = max(risk, atr)

    if final_risk <= 0:
        return 10.0 # High RR if no projected risk

    return reward / final_risk

def get_smart_targets(entry_price, side, atr, confidence, fcast_data=None):
    """
    Expert Intelligence TP/SL Engine (Enhanced with Echo Forecast).
    Uses ATR and Projected Market Structure to set optimal targets.
    """
    if atr == 0:
        return None, None

    is_long = side == 'long'

    # 1. Base ATR Risk (1.5x ATR for SL)
    sl_dist = 1.5 * atr

    # 2. Echo Structure Alignment
    # If forecast shows a clear structure peak/trough, we use it to cap or extend TP.
    fcast_tp_dist = 0
    if fcast_data and 'correlation' in fcast_data and fcast_data['correlation'] > 0.6:
        fcast_high = fcast_data.get('high')
        fcast_low = fcast_data.get('low')

        if is_long and fcast_high:
            fcast_tp_dist = fcast_high - entry_price
        elif not is_long and fcast_low:
            fcast_tp_dist = entry_price - fcast_low

    # 3. Dynamic Risk Reward (2x to 5x base risk)
    rr = 2 + (3 * (confidence / 100))
    tp_dist = sl_dist * rr

    # If Echo projects a larger move with high confidence, we let it run
    if fcast_tp_dist > tp_dist:
        tp_dist = fcast_tp_dist

    tp_price = (entry_price + tp_dist) if is_long else (entry_price - tp_dist)
    sl_price = (entry_price - sl_dist) if is_long else (entry_price + sl_dist)

    return tp_price, sl_price

def calculate_echo_forecast(df, eval_window=50, forecast_window=50, fmode='Similarity', projection='Pattern'):
    """
    Expert Intelligence Echo Forecast (Simplified LuxAlgo Port).
    Identifies historical fractal similarities and projects price action.
    """
    if df is None or len(df) < (eval_window + forecast_window * 2 + 1):
        return None, 0

    src = df['close'].values
    deltas = df['close'].diff().values
    
    ref = src[-forecast_window:]
    
    best_val = -1.0 if fmode == 'Similarity' else 1.0
    best_k = 0

    for i in range(eval_window):
        match_end_idx = len(src) - forecast_window - i
        match_start_idx = match_end_idx - forecast_window

        if match_start_idx < 0:
            break

        b = src[match_start_idx:match_end_idx]

        # Pearson Correlation
        std_ref = np.std(ref)
        std_b = np.std(b)

        if std_ref == 0 or std_b == 0:
            r = 0
        else:
            r = np.corrcoef(ref, b)[0, 1]

        if np.isnan(r): r = 0

        if fmode == 'Similarity':
            if r > best_val:
                best_val = r
                best_k = i
        else: # Dissimilarity
            if r < best_val:
                best_val = r
                best_k = i

    # Identify the data window to project
    if projection == 'Pattern':
        # Replays the moves WITHIN the matched historical window
        match_start = len(src) - forecast_window*2 - best_k
        forecast_deltas = deltas[match_start : match_start + forecast_window]
    else:
        # Projects the moves that FOLLOWED the matched historical window
        match_end = len(src) - forecast_window - best_k
        forecast_deltas = deltas[match_end : match_end + forecast_window]

    # Construction base
    current_price = src[-1]
    forecast_prices = []
    temp_price = current_price

    for d in forecast_deltas:
        if np.isnan(d): d = 0
        temp_price += d
        forecast_prices.append(temp_price)

    return forecast_prices, best_val
