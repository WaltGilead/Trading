import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

logger = logging.getLogger(__name__)

class SentimentEngine:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
    def analyze_text_sentiment(self, text):
        """
        Returns a compound sentiment score between -1 and 1.
        """
        if not text:
            return 0.0
        scores = self.analyzer.polarity_scores(text)
        return scores['compound']

    def compute_news_score(self, news_items):
        """
        Aggregates sentiment from news headlines.
        Returns a score 0-100 and a summary string.
        """
        if not news_items:
            return 50, "No news found."
            
        total_score = 0
        count = 0
        headlines = []
        
        for item in news_items:
            title = item.get('title')
            if title:
                score = self.analyze_text_sentiment(title)
                total_score += score
                count += 1
                headlines.append(title)
                
        if count == 0:
            return 50, "No valid headlines."
            
        avg_score = total_score / count
        # Normalize -1 to 1 -> 0 to 100
        normalized_score = (avg_score + 1) * 50
        
        # Simple summary: top 3 headlines
        summary = " | ".join(headlines[:3])
        
        return round(normalized_score, 2), summary

    def compute_economic_score(self, econ_data):
        """
        Computes economic score based on FRED data.
        econ_data: dict of DataFrames (CPI, Unemployment, FedFunds)
        """
        score = 50
        details = []
        
        if not econ_data:
            return 50, "No Data"

        # 1. Unemployment (Lower is generally bullish for economy, but too low = inflation risk. Rising = Recession)
        unemp = econ_data.get('Unemployment')
        if unemp is not None and not unemp.empty:
            curr_unemp = unemp.iloc[-1].values[0]
            prev_unemp = unemp.iloc[-2].values[0]
            
            if curr_unemp < prev_unemp:
                score += 5
                details.append("Unemployment Falling")
            elif curr_unemp > prev_unemp:
                score -= 5
                details.append("Unemployment Rising")
            
            # Absolute levels
            if curr_unemp < 4.0:
                score += 5 # Strong labor market
            elif curr_unemp > 6.0:
                score -= 10 # Weak labor market

        # 2. CPI (Inflation) - Lower/Stable is bullish
        cpi = econ_data.get('CPI')
        if cpi is not None and not cpi.empty:
            # Calculate YoY Inflation
            try:
                curr_cpi = cpi.iloc[-1].values[0]
                prev_year_cpi = cpi.iloc[-13].values[0] # 12 months ago
                inflation_rate = ((curr_cpi - prev_year_cpi) / prev_year_cpi) * 100
                
                if inflation_rate > 3.0:
                    score -= 10
                    details.append(f"High Inflation ({inflation_rate:.1f}%)")
                elif inflation_rate < 2.0:
                    score += 5
                    details.append("Stable Inflation")
            except:
                pass

        # 3. Fed Funds (Interest Rates) - Lower is bullish
        fed = econ_data.get('FedFunds')
        if fed is not None and not fed.empty:
            curr_rate = fed.iloc[-1].values[0]
            if curr_rate > 5.0:
                score -= 10
                details.append("High Rates")
            elif curr_rate < 2.0:
                score += 10
                details.append("Low Rates")

        score = max(0, min(100, score))
        label = ", ".join(details) if details else "Neutral"
        
        return score, label

    def compute_macro_score(self, market_data):
        """
        Computes macro score based on VIX, DXY, and US10Y (if available).
        Lower VIX/DXY is generally bullish for risk assets.
        Returns score 0-100 and label.
        """
        score = 50
        
        # VIX Logic
        vix_data = market_data.get('VIX')
        if vix_data is not None and not vix_data.empty:
            try:
                # Handle potential MultiIndex or Series
                current_vix = vix_data['Close'].iloc[-1]
                if isinstance(current_vix, pd.Series):
                    current_vix = current_vix.item()
                
                # VIX < 15 Bullish, > 25 Bearish
                if current_vix < 15:
                    score += 10
                elif current_vix > 25:
                    score -= 10
                elif current_vix > 35:
                    score -= 20
            except Exception as e:
                logger.error(f"Error calculating VIX score: {e}")
                
        # DXY Logic
        dxy_data = market_data.get('DXY')
        if dxy_data is not None and not dxy_data.empty:
            try:
                current_dxy = dxy_data['Close'].iloc[-1]
                if isinstance(current_dxy, pd.Series):
                    current_dxy = current_dxy.item()
                    
                # DXY < 95 Bullish, > 105 Bearish
                if current_dxy < 95:
                    score += 10
                elif current_dxy > 105:
                    score -= 10
            except Exception as e:
                logger.error(f"Error calculating DXY score: {e}")
        
        # Clamp score
        score = max(0, min(100, score))
        
        label = "Neutral"
        if score >= 60:
            label = "Bullish"
        elif score <= 40:
            label = "Bearish"
            
        return score, label

if __name__ == "__main__":
    # Test
    engine = SentimentEngine()
    print(f"Sentiment of 'Bitcoin is crashing hard!': {engine.analyze_text_sentiment('Bitcoin is crashing hard!')}")
    print(f"Sentiment of 'US economy is booming': {engine.analyze_text_sentiment('US economy is booming')}")
    
    # Mock data for macro
    mock_data = {
        'VIX': pd.DataFrame({'Close': [12.0]}),
        'DXY': pd.DataFrame({'Close': [102.0]})
    }
    print(f"Macro Score (Low VIX): {engine.compute_macro_score(mock_data)}")
