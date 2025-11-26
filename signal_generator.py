import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SignalGenerator:
    def __init__(self):
        pass

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        df['MACD_Line'] = df['Close'].ewm(span=fast, adjust=False).mean() - df['Close'].ewm(span=slow, adjust=False).mean()
        df['MACD_Signal'] = df['MACD_Line'].ewm(span=signal, adjust=False).mean()
        df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']
        return df

    def calculate_bollinger_bands(self, df, window=20, num_std=2):
        df['BB_Mid'] = df['Close'].rolling(window=window).mean()
        df['BB_Std'] = df['Close'].rolling(window=window).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * num_std)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * num_std)
        return df

    def calculate_obv(self, df):
        """
        Calculates On-Balance Volume (OBV).
        """
        if 'Volume' not in df.columns:
            df['OBV'] = 0
            return df
            
        # If Close > Prev Close, Volume is added. If Close < Prev Close, Volume is subtracted.
        # We use numpy sign for efficiency
        change = df['Close'].diff()
        direction = np.sign(change)
        direction.iloc[0] = 0 # First element is NaN/0
        
        # Volume flow
        volume_flow = direction * df['Volume']
        df['OBV'] = volume_flow.cumsum()
        
        # Add OBV Slope (5-period) to detect trend
        df['OBV_Slope'] = df['OBV'].diff(5)
        return df

    def calculate_adx(self, df, period=14):
        """
        Calculates Average Directional Index (ADX) to measure trend strength.
        """
        if df is None or df.empty:
            return df
            
        # True Range
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        
        # Directional Movement
        df['UpMove'] = df['High'] - df['High'].shift(1)
        df['DownMove'] = df['Low'].shift(1) - df['Low']
        
        df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
        df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
        
        # Smooth TR and DM
        # Wilder's Smoothing (alpha = 1/n) is standard, but EWMA is close enough for this agent
        df['TR_Smooth'] = df['TR'].rolling(window=period).mean() # Simple moving average for stability
        df['+DM_Smooth'] = df['+DM'].rolling(window=period).mean()
        df['-DM_Smooth'] = df['-DM'].rolling(window=period).mean()
        
        # DI
        df['+DI'] = 100 * (df['+DM_Smooth'] / df['TR_Smooth'])
        df['-DI'] = 100 * (df['-DM_Smooth'] / df['TR_Smooth'])
        
        # DX
        df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
        
        # ADX
        df['ADX'] = df['DX'].rolling(window=period).mean()
        
        # Cleanup temp columns
        df.drop(['H-L', 'H-PC', 'L-PC', 'TR', 'UpMove', 'DownMove', '+DM', '-DM', 'TR_Smooth', '+DM_Smooth', '-DM_Smooth', 'DX'], axis=1, inplace=True)
        
        return df

    def calculate_stoch_rsi(self, df, period=14, k_period=3, d_period=3):
        rsi = self.calculate_rsi(df['Close'], period)
        
        rsi_min = rsi.rolling(window=period).min()
        rsi_max = rsi.rolling(window=period).max()
        
        # Avoid division by zero
        denominator = rsi_max - rsi_min
        denominator = denominator.replace(0, 1e-10)
        
        stoch_rsi = (rsi - rsi_min) / denominator
        
        df['StochRSI_K'] = stoch_rsi.rolling(window=k_period).mean() * 100
        df['StochRSI_D'] = df['StochRSI_K'].rolling(window=d_period).mean()
        return df

    def calculate_technical_indicators(self, df):
        if df is None or df.empty:
            return df
        df = df.copy()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df = self.calculate_macd(df)
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_stoch_rsi(df)
        df = self.calculate_obv(df)
        df = self.calculate_adx(df)
        return df

    def generate_signal(self, ticker, multi_timeframe_data, sentiment_scores, ml_context=None):
        """
        Generates a trading signal based on multi-timeframe technicals, sentiment, and ML.
        multi_timeframe_data: {'1h': df, '1d': df}
        ml_context: {'prediction': float, 'confidence': float} (Optional)
        """
        df_1d = multi_timeframe_data.get('1d', {}).get(ticker)
        df_1h = multi_timeframe_data.get('1h', {}).get(ticker)

        if df_1d is None or df_1d.empty:
            return "NEUTRAL", "Neutral", 0, "Insufficient Data"

        # Calculate Indicators
        df_1d = self.calculate_technical_indicators(df_1d)
        latest_1d = df_1d.iloc[-1]
        
        latest_1h = None
        if df_1h is not None and not df_1h.empty:
            df_1h = self.calculate_technical_indicators(df_1h)
            latest_1h = df_1h.iloc[-1]

        # Sentiment
        macro_score = sentiment_scores.get('macro', 50)
        news_score = sentiment_scores.get('news', 50)
        econ_score = sentiment_scores.get('economic', 50)
        avg_sentiment = (macro_score + news_score + econ_score) / 3

        # Logic - Using weighted scoring system
        signal = "NEUTRAL"
        bias = "Neutral"
        confidence = 50
        reasons = []
        
        # Score accumulator (-100 to +100)
        bullish_score = 0
        bearish_score = 0

        # 1. Trend (Daily) - Weight: 25 points
        trend_bullish = latest_1d['Close'] > latest_1d['SMA_200']
        if trend_bullish:
            bullish_score += 25
            reasons.append("Daily Trend: Bullish (Above SMA 200)")
        else:
            bearish_score += 25
            reasons.append("Daily Trend: Bearish (Below SMA 200)")

        # 2. Momentum (Hourly) - Weight: 20 points
        mom_bullish = False
        if latest_1h is not None:
            macd_bullish = latest_1h['MACD_Line'] > latest_1h['MACD_Signal']
            rsi_bullish = latest_1h['RSI'] > 50
            
            if macd_bullish and rsi_bullish:
                mom_bullish = True
                bullish_score += 20
                reasons.append("Hourly Momentum: Bullish (MACD+RSI aligned)")
            elif macd_bullish or rsi_bullish:
                # Partial bullish signal
                bullish_score += 10
                reasons.append("Hourly Momentum: Mixed (Partial bullish signals)")
            else:
                bearish_score += 20
                reasons.append("Hourly Momentum: Bearish")
        
        # 3. Sentiment - Weight: 25 points
        if avg_sentiment > 60:
            sentiment_strength = min((avg_sentiment - 60) * 0.8, 25)  # Scale to 25 max
            bullish_score += sentiment_strength
            reasons.append(f"Sentiment: Bullish ({avg_sentiment:.0f}/100)")
        elif avg_sentiment < 40:
            sentiment_strength = min((40 - avg_sentiment) * 0.8, 25)
            bearish_score += sentiment_strength
            reasons.append(f"Sentiment: Bearish ({avg_sentiment:.0f}/100)")
        else:
            reasons.append(f"Sentiment: Neutral ({avg_sentiment:.0f}/100)")
        
        # 4. RSI Overbought/Oversold - Weight: 10 points
        daily_rsi = latest_1d['RSI']
        if daily_rsi > 70:
            bearish_score += 10
            reasons.append(f"RSI Overbought ({daily_rsi:.1f})")
        elif daily_rsi < 30:
            bullish_score += 10
            reasons.append(f"RSI Oversold ({daily_rsi:.1f})")
            
        # 5. Trend Strength (ADX) - Weight: 10 points (Filter)
        adx = latest_1d.get('ADX', 0)
        if adx > 25:
            if trend_bullish:
                bullish_score += 10
                reasons.append(f"Strong Trend (ADX: {adx:.1f})")
            else:
                bearish_score += 10
                reasons.append(f"Strong Downtrend (ADX: {adx:.1f})")
        else:
            reasons.append(f"Weak Trend (ADX: {adx:.1f})")
            
        # 6. Volume Confirmation (OBV) - Weight: 10 points
        obv_slope = latest_1d.get('OBV_Slope', 0)
        if obv_slope > 0:
            bullish_score += 10
            reasons.append("Volume: Accumulation (Rising OBV)")
        elif obv_slope < 0:
            bearish_score += 10
            reasons.append("Volume: Distribution (Falling OBV)")
            
        # 7. ML Prediction - Weight: 15 points
        if ml_context:
            pred = ml_context.get('prediction', 0)
            conf = ml_context.get('confidence', 0)
            
            if conf > 60:
                if pred > 0.001: # > 0.1% return
                    bullish_score += 15
                    reasons.append(f"ML Signal: Bullish (Pred: {pred*100:.2f}%)")
                elif pred < -0.001: # < -0.1% return
                    bearish_score += 15
                    reasons.append(f"ML Signal: Bearish (Pred: {pred*100:.2f}%)")
        
        # 8. Volatility (Bollinger Bands - Daily) - Bonus
        bb_squeeze = (latest_1d['BB_Upper'] - latest_1d['BB_Lower']) / latest_1d['BB_Mid'] < 0.05
        if bb_squeeze:
            reasons.append("** Volatility Squeeze (Breakout likely)")

        # Determine Signal based on score differential
        net_score = bullish_score - bearish_score
        
        if net_score >= 40:
            signal = "BUY"
            bias = "Bullish"
            confidence = min(60 + abs(net_score) // 2, 95)
            reasons.append(f">> Strong Bullish Alignment (Score: +{net_score:.0f})")
        elif net_score >= 15:
            signal = "NEUTRAL"
            bias = "Bullish"
            confidence = 55
            reasons.append(f"Bullish Bias (Score: +{net_score:.0f})")
        elif net_score <= -40:
            signal = "SELL"
            bias = "Bearish"
            confidence = min(60 + abs(net_score) // 2, 95)
            reasons.append(f">> Strong Bearish Alignment (Score: {net_score:.0f})")
        elif net_score <= -15:
            signal = "NEUTRAL"
            bias = "Bearish"
            confidence = 55
            reasons.append(f"Bearish Bias (Score: {net_score:.0f})")
        else:
            confidence = 50
            reasons.append(f"Neutral Market (Score: {net_score:+.0f})")

        explanation = " | ".join(reasons)
        return signal, bias, confidence, explanation

if __name__ == "__main__":
    # Test
    import numpy as np
    
    gen = SignalGenerator()
    
    # Mock Data - Need enough data for 200 SMA + ADX smoothing
    dates = pd.date_range(start='2023-01-01', periods=300)
    close = np.linspace(100, 150, 300) # Uptrend
    # Add some noise
    close += np.random.normal(0, 1, 300)
    
    high = close + 2
    low = close - 2
    volume = np.random.randint(1000, 5000, 300)
    
    df = pd.DataFrame({
        'Close': close, 
        'High': high, 
        'Low': low, 
        'Volume': volume
    }, index=dates)
    
    scores = {'macro': 70, 'news': 65, 'economic': 50} # Bullish
    
    # Pass as dictionary as expected by generate_signal
    # Structure: {'1d': {'TEST': df}, '1h': {'TEST': df}}
    data_map = {
        '1d': {'TEST': df},
        '1h': {'TEST': df}
    }
    sig, bias, conf, exp = gen.generate_signal("TEST", data_map, scores)
    print(f"Signal: {sig}, Bias: {bias}, Conf: {conf}")
    print(f"Explanation: {exp}")
